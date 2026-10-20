"""Coded scVI module: K information bits protected by a repetition code, on top of the DVAE++
smoothing transform.

This is the "coded" model of the paper. CodedVAESmoothing (_module.py) is the same latent without
the code (the "uncoded" ablation, r = 1).
"""

from typing import Dict

import torch
from scvi import REGISTRY_KEYS
from scvi.module import VAE
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import Bernoulli, kl_divergence

from ._module import (
    BinaryLatentEncoder,
    _log_smoothing_density,
    _marginal_ll_with_prior,
    _SmoothingPosterior,
    dclamp,
    sample_from_qz_given_x,
)

TensorDict = Dict[str, torch.Tensor]


def _build_repetition_code(n_info_bits: int, rep_factor: int) -> torch.Tensor:
    """Rate 1/rep_factor repetition code generator matrix G, shape (n_info_bits, n_bits).

    Codeword bit c_j = information bit m_{j // rep_factor}, i.e. each info bit is repeated
    `rep_factor` times contiguously: G[i, i*rep_factor:(i+1)*rep_factor] = 1.
    """
    n_bits = n_info_bits * rep_factor
    G = torch.zeros(n_info_bits, n_bits)
    for i in range(n_info_bits):
        G[i, i * rep_factor : (i + 1) * rep_factor] = 1.0
    return G


class CodedVAEECC(VAE):
    """CodedVI + repetition-code ECC on top of the DVAE++ / Coded-DVAE smoothing latent.

    n_info_bits information bits m are each protected by repeating them rep_factor times,
    forming a codeword c of length n_bits = n_info_bits * rep_factor (this is the actual
    z dimensionality fed to the decoder). The bit_encoder still predicts q(c_j=1|x)
    independently per codeword position (same BinaryLatentEncoder architecture as
    CodedVAESmoothing, just with n_bits = codeword length instead of n_info_bits).

    A soft-decoding step then recovers the underlying information-bit posterior by
    naive-Bayes combination (product in probability space, sum in log space) of the
    rep_factor replicate log-likelihoods for each info bit -- ported from the "rep"
    inference path of mariamartinezgarcia/codedVAE's forward()/get_elbo_rep(). The
    resulting (less noisy) info-bit posterior is re-broadcast to all codeword positions
    before applying the same DVAE++ exponential-smoothing reparameterization used by
    CodedVAESmoothing (sample_from_qz_given_x, imported unchanged from _module.py).

    The KL term is computed on the n_info_bits marginals only (not the n_bits codeword
    positions), since codeword bits are a deterministic function of the info bits and
    carry no extra degrees of freedom -- matching get_elbo_rep's kl_div_bernoulli call
    in the reference implementation.
    """

    def __init__(
        self,
        n_input: int,
        n_info_bits: int = 16,
        rep_factor: int = 4,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        beta: float = 10.0,
        **kwargs,
    ):
        n_bits = n_info_bits * rep_factor
        super().__init__(n_input, n_latent=n_bits, n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate, **kwargs)
        self.n_info_bits = n_info_bits
        self.rep_factor = rep_factor
        self.n_bits = n_bits
        # VAE does not store n_hidden as an attribute; keep our own copy for bit_encoder
        self.n_hidden = n_hidden
        cat_list = [self.n_batch] if self.encode_covariates else []
        self.bit_encoder = BinaryLatentEncoder(
            n_input, n_bits, cat_list, n_hidden=self.n_hidden, n_layers=n_layers, dropout_rate=dropout_rate
        )

        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("prior_m_logits", torch.zeros(n_info_bits))  # Bernoulli(0.5)

        G = _build_repetition_code(n_info_bits, rep_factor)
        self.register_buffer("G", G)  # (n_info_bits, n_bits)
        self.register_buffer("H", G.t())  # (n_bits, n_info_bits)

        # same closed-form mean/variance of r(z|m=0) as CodedVAESmoothing (see that class
        # for the derivation); used only for the "qz" Normal proxy that
        # VAEMixin.get_latent_representation requires (qz.loc/qz.scale).
        beta_t = self.beta
        mu0 = 1.0 / beta_t - 1.0 / (torch.exp(beta_t) - 1.0)
        m2_0 = (
            -torch.exp(-beta_t)
            - 2 * torch.exp(-beta_t) / beta_t
            + 2 * (1 - torch.exp(-beta_t)) / beta_t**2
        ) / (1 - torch.exp(-beta_t))
        self.register_buffer("mu0", mu0)
        self.register_buffer("var0", m2_0 - mu0**2)

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        out = super().inference(x, batch_index, cont_covs, cat_covs, n_samples)
        # reuse the library computed by the parent; only replace the z-related parts

        x_ = torch.log1p(x) if self.log_variational else x
        cat_list = (batch_index,) if self.encode_covariates else ()
        c_logits = self.bit_encoder(x_, *cat_list)  # (batch, n_bits) per-codeword-position logits
        c_probs = torch.sigmoid(c_logits)
        c_probs = dclamp(c_probs, 1e-6, 1 - 1e-6)  # avoid log(0) below

        # Soft-decode: combine the rep_factor replicate log-likelihoods per info bit
        # (naive-Bayes / product-of-experts, assuming replicate positions are
        # conditionally independent given the info bit -- standard repetition-code
        # soft decoding).
        logpm1 = torch.log(c_probs) @ self.H
        logpm0 = torch.log(1 - c_probs) @ self.H
        log_marginals = torch.stack([logpm0, logpm1], dim=-1)
        log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
        m_logits = log_marginals_norm[..., 1] - log_marginals_norm[..., 0]  # (batch, n_info_bits)
        m_probs = log_marginals_norm[..., 1].exp()

        # Re-encode: broadcast the (less noisy) info-bit posterior back out to all
        # codeword positions before sampling z.
        qc = m_probs @ self.G  # (batch, n_bits)

        z = sample_from_qz_given_x(qc, self.beta, n_samples=n_samples)  # (batch, n_bits, n_samples)
        z = z.squeeze(-1) if n_samples == 1 else z.permute(2, 0, 1)  # (batch, n_bits) or (n_samples, batch, n_bits)

        qzm = self.mu0 * (1 - qc) + (1 - self.mu0) * qc
        qzv = (qc**2 + (1 - qc) ** 2) * self.var0
        # overwrite the parent's stale "qz" (see CodedVAE.inference for why this is needed).
        # logit(qc) = m_logits @ G exactly (G has one 1 per column, i.e. a pure broadcast/copy,
        # so sigmoid commutes with it) -- gives _SmoothingPosterior the correct per-codeword-
        # position mixture weight to compute the true (non-Gaussian) q(z|x) density.
        qz = _SmoothingPosterior(qzm, qzv.sqrt(), m_logits @ self.G, self.beta)

        # Keep the pre-decoding codeword logits as well.  They are deliberately distinct
        # from ``m_logits``: the latter is the K-dimensional, soft-decoded information
        # word used by the ELBO, while ``c_logits`` is the N-dimensional encoder output
        # needed to measure how much the repetition decoder recovers under perturbation.
        out.update(
            {"z": z, "qz": qz, "qzm": qzm, "qzv": qzv, "m_logits": m_logits, "c_logits": c_logits}
        )
        return out

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        m_logits = inference_outputs["m_logits"]

        # KL is over the n_info_bits marginals only: codeword bits are a deterministic
        # function of the info bits (via G) and carry no extra degrees of freedom.
        kl_m = kl_divergence(
            Bernoulli(logits=m_logits),
            Bernoulli(logits=self.prior_m_logits.expand_as(m_logits)),
        ).sum(-1)

        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.use_observed_lib_size:
            kl_l = torch.zeros_like(kl_m)
        else:
            kl_l = kl_divergence(inference_outputs["ql"], generative_outputs["pl"]).sum(dim=-1)

        recon = -generative_outputs["px"].log_prob(x).sum(-1)

        loss = torch.mean(recon + kl_weight * kl_m + kl_l)
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon,
            kl_local={"kl_m": kl_m, "kl_l": kl_l},
        )

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples, return_mean=False, n_mc_samples_per_pass=1):
        """Same as scvi.module.VAE.marginal_ll, but the prior p(z) is the true DVAE++
        smoothing-transform mixture at weight 0.5, not the parent's hardcoded standard-normal
        N(0,1) assumption (which does not hold on [0,1]). q(z|x) is likewise fixed via
        _SmoothingPosterior.log_prob, set up in inference().
        """

        def log_p_z(z):
            zero_logit = torch.zeros((), device=z.device, dtype=z.dtype)
            return _log_smoothing_density(z, zero_logit, self.beta).sum(dim=-1)

        return _marginal_ll_with_prior(self, tensors, n_mc_samples, log_p_z, return_mean, n_mc_samples_per_pass)
