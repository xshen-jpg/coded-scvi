import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from ._module import CodedVAE, CodedVAESmoothing
from ._module_ecc import CodedVAEECC

logger = logging.getLogger(__name__)


class CODEDVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """CodedVI: scVI with a binary (Bernoulli-gated Gaussian mixture) latent space (not used in the paper).

    c = m in this phase; no ECC codeword constraint is applied yet.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~codedvi.CODEDVI.setup_anndata`.
    n_bits
        Number of latent bits (K).
    n_hidden
        Number of nodes per hidden layer.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~codedvi.CodedVAE`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> codedvi.CODEDVI.setup_anndata(adata, batch_key="batch")
    >>> vae = codedvi.CODEDVI(adata, n_bits=32)
    >>> vae.train()
    >>> adata.obsm["X_codedvi"] = vae.get_latent_representation()
    >>> adata.obsm["pi_codedvi"] = vae.get_bit_probabilities()
    """

    _module_cls = CodedVAE

    def __init__(
        self,
        adata: AnnData,
        n_bits: int = 32,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_bits=n_bits,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"CODEDVI (no ECC) model with n_bits: {n_bits}, n_hidden: {n_hidden}, "
            f"n_layers: {n_layers}, dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}."
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.

        `batch_key=None` still registers a constant-valued batch field: the batch_index
        pipeline is kept end-to-end (module, inference/generative inputs, registry) even
        though it degenerates to a single constant column for a single-batch dataset.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_bit_probabilities(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return q(m=1|x), shape (n_cells, n_bits).

        For the coded models these are the soft-decoded probabilities of the information bits,
        the readable cell representation.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        probs = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            probs.append(torch.sigmoid(outputs["m_logits"]).cpu())
        return torch.cat(probs).numpy()


class CODEDVISmoothing(CODEDVI):
    """Uncoded scVI: binary latent with the fixed DVAE++ exponential-smoothing transform, no code.

    Parallel to :class:`CODEDVI` (the learnable Gaussian-mixture latent) for direct ELBO/bit-
    collapse comparison; shares the same encoder/decoder architecture and training pipeline
    (:class:`~scvi.model.base.UnsupervisedTrainingMixin`), and differs only in the module used
    for z, see :class:`~codedvi._module.CodedVAESmoothing`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~codedvi.CODEDVISmoothing.setup_anndata`.
    n_bits
        Number of latent bits (K).
    n_hidden
        Number of nodes per hidden layer.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    beta
        Fixed temperature of the exponential smoothing transformation (not learned).
        Default 10, matching mariamartinezgarcia/codedVAE.
    **model_kwargs
        Keyword args for :class:`~codedvi._module.CodedVAESmoothing`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> codedvi.CODEDVISmoothing.setup_anndata(adata, batch_key="batch")
    >>> vae = codedvi.CODEDVISmoothing(adata, n_bits=32)
    >>> vae.train()
    >>> adata.obsm["X_codedvi_smoothing"] = vae.get_latent_representation()
    >>> adata.obsm["pi_codedvi_smoothing"] = vae.get_bit_probabilities()
    """

    _module_cls = CodedVAESmoothing


class CODEDVIECC(CODEDVI):
    """Coded scVI: binary latent protected by a repetition code, on the DVAE++ smoothing latent.

    Third leg of the 3-way comparison alongside :class:`CODEDVI` (Gaussian-mixture) and
    :class:`CODEDVISmoothing` (fixed exponential smoothing, no ECC). n_info_bits information
    bits are each protected by a rate-1/rep_factor repetition code, so the actual z
    dimensionality fed to the decoder is n_info_bits * rep_factor. See
    :class:`~codedvi._module_ecc.CodedVAEECC` for the soft-decoding math.

    Does not reuse :meth:`CODEDVI.__init__`: that hardcodes ``n_bits`` as the module's
    latent-dim kwarg, which :class:`~codedvi._module_ecc.CodedVAEECC` does not accept (it
    derives its codeword length from ``n_info_bits * rep_factor`` instead).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~codedvi.CODEDVIECC.setup_anndata`.
    n_info_bits
        Number of *information* bits (K); the actual latent dimensionality is
        ``n_info_bits * rep_factor``.
    rep_factor
        Repetition factor of the code (each info bit repeated this many times).
    n_hidden
        Number of nodes per hidden layer.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    beta
        Fixed temperature of the exponential smoothing transformation (not learned).
    **model_kwargs
        Keyword args for :class:`~codedvi._module_ecc.CodedVAEECC`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> codedvi.CODEDVIECC.setup_anndata(adata, batch_key="batch")
    >>> vae = codedvi.CODEDVIECC(adata, n_info_bits=16, rep_factor=4)
    >>> vae.train()
    >>> adata.obsm["X_codedvi_ecc"] = vae.get_latent_representation()
    >>> adata.obsm["pi_codedvi_ecc"] = vae.get_bit_probabilities()
    """

    _module_cls = CodedVAEECC

    def __init__(
        self,
        adata: AnnData,
        n_info_bits: int = 16,
        rep_factor: int = 4,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        beta: float = 10.0,
        **model_kwargs,
    ):
        BaseModelClass.__init__(self, adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_info_bits=n_info_bits,
            rep_factor=rep_factor,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            beta=beta,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"CODEDVIECC (repetition code, n_info_bits: {n_info_bits}, rep_factor: {rep_factor}) "
            f"model with n_hidden: {n_hidden}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"dispersion: {dispersion}, gene_likelihood: {gene_likelihood}."
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @torch.inference_mode()
    def get_codeword_probabilities(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return the raw N-dimensional encoder probabilities ``q^u(c_j=1|x)``.

        Unlike :meth:`get_bit_probabilities`, this intentionally exposes the predictions
        *before* repetition-code soft decoding.  It is principally useful for evaluating
        whether decoding improves recovery after an input perturbation.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        probs = []
        for tensors in scdl:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            probs.append(torch.sigmoid(outputs["c_logits"]).cpu())
        return torch.cat(probs).numpy()

    def get_hard_information_bits(self, **kwargs) -> np.ndarray:
        """Return MAP information bits, with shape ``(n_cells, n_info_bits)``."""
        return (self.get_bit_probabilities(**kwargs) >= 0.5).astype(np.int8)

    def get_hard_codewords(self, **kwargs) -> np.ndarray:
        """Hard-encode MAP information bits into valid repetition-code codewords."""
        info = self.get_hard_information_bits(**kwargs)
        return (info @ self.module.G.detach().cpu().numpy()).astype(np.int8)
