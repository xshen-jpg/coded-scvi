"""Model wrappers for the uncoded/coded MrVI variants.

TorchMRVI.__init__ is already generic over `_module_cls` (it just forwards `**model_kwargs` to
`self._module_cls(n_input=..., n_sample=..., n_batch=..., n_labels=..., n_obs_per_sample=...,
**model_kwargs)`), so no `__init__` override is needed here -- unlike codedvi.CODEDVIECC, which
had to override `BaseModelClass.__init__` because `CODEDVI.__init__` hardcoded an incompatible
module kwarg name.
"""

import numpy as np
import torch
from scvi.external.mrvi_torch import TorchMRVI

from ._module import CodedMRVAEECC, CodedMRVAESmoothing


class _BitProbabilitiesMixin:
    """Shared get_bit_probabilities() for CodedMRVI/CodedMRVIECC.

    Both modules' inference() stores "m_logits" -- CodedMRVAESmoothing's per-bit logits
    (n_bits dims, no ECC), CodedMRVAEECC's *soft-decoded info-bit* marginal logits (n_info_bits
    dims, the denoised signal, not the raw n_bits-dim codeword) -- same convention as
    codedvi.CODEDVI.get_bit_probabilities.
    """

    @torch.inference_mode()
    def get_bit_probabilities(self, adata=None, indices=None, batch_size=None) -> np.ndarray:
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        probs = []
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            probs.append(torch.sigmoid(outputs["m_logits"]).cpu())
        return torch.cat(probs).numpy()


class CodedMRVI(_BitProbabilitiesMixin, TorchMRVI):
    """MrVI with `u` produced by the DVAE++ smoothing transform (uncoded binary latent)."""

    _module_cls = CodedMRVAESmoothing


class CodedMRVIECC(_BitProbabilitiesMixin, TorchMRVI):
    """MrVI with `u` produced by the smoothing transform + a repetition-code ECC layer."""

    _module_cls = CodedMRVAEECC

    @torch.inference_mode()
    def get_codeword_probabilities(self, adata=None, indices=None, batch_size=None) -> np.ndarray:
        """Return raw N-dimensional codeword probabilities before soft decoding."""
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        probs = []
        for tensors in dataloader:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            probs.append(torch.sigmoid(outputs["c_logits"]).cpu())
        return torch.cat(probs).numpy()

    def get_hard_information_bits(self, **kwargs) -> np.ndarray:
        """Return MAP information bits, shape ``(n_cells, n_info_bits)``."""
        return (self.get_bit_probabilities(**kwargs) >= 0.5).astype(np.int8)

    def get_hard_codewords(self, **kwargs) -> np.ndarray:
        """Hard-encode MAP information bits into valid repetition-code codewords."""
        info = self.get_hard_information_bits(**kwargs)
        return (info @ self.module.G.detach().cpu().numpy()).astype(np.int8)
