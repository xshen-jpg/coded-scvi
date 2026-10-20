from typing import Callable, Dict

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.module import VAE
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import FCLayers
from torch import logsumexp, nn
from torch.distributions import Bernoulli, Normal, kl_divergence
from torch.nn.functional import logsigmoid

TensorDict = Dict[str, torch.Tensor]


class BinaryLatentEncoder(nn.Module):
    """Encodes x into bit logits only; produces no x-dependent component parameters."""

    def __init__(
        self,
        n_input: int,
        n_bits: int,
        n_cat_list,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.m_logits = nn.Linear(n_hidden, n_bits)

    def forward(self, x, *cat_list):
        h = self.backbone(x, *cat_list)
        return self.m_logits(h)


class _DifferentiableClamp(torch.autograd.Function):
    """Clamp with gradient 1 everywhere (identity backward), avoiding dead gradients at the bounds.

    Ported from mariamartinezgarcia/codedVAE (src/nn/modules.py), used by
    sample_from_qz_given_x to keep q_i away from 1 without blocking gradient flow there.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    return _DifferentiableClamp.apply(input, min, max)


def sample_from_qz_given_x(qi: torch.Tensor, beta: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """DVAE++ inverse-CDF reparameterization for the exponential smoothing transformation.

    Ported line-for-line (including the dclamp/b/c/dif numerical-stability handling) from
    mariamartinezgarcia/codedVAE (src/utils/sampling.py::sample_from_qz_given_x).

    Parameters
    ----------
    qi
        q(bit=1|x), shape (batch, n_bits).
    beta
        Fixed temperature controlling the decay of the smoothing exponentials.
    n_samples
        Number of z samples to draw per bit.

    Returns
    -------
    z of shape (batch, n_bits, n_samples).
    """
    epsilon = 1e-6

    qi = qi.unsqueeze(2).repeat(1, 1, n_samples)
    ones = torch.ones_like(qi)

    # Clamp to avoid divisions by 0 in the reparameterization
    qi = dclamp(qi, 0, 1 - 1e-3)

    # Sample from U(0,1)
    rho = torch.rand_like(qi)

    # Reparameterization
    b = (rho + torch.exp(-beta) * (qi - rho)) / (ones - qi) - ones
    c = -(qi * torch.exp(-beta)) / (ones - qi)

    dif = torch.sqrt(torch.pow(b, 2) - 4 * c) - b
    dif = torch.where(dif <= 0, torch.full_like(dif, epsilon), dif)  # avoid negative/zero from numerical imprecision

    z = (-1 / beta) * torch.log(dif / 2)
    return z


def _log_smoothing_density(z: torch.Tensor, logit_weight: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Per-element log[q*r(z|m=1) + (1-q)*r(z|m=0)], q = sigmoid(logit_weight).

    r(z|m=0) = beta*exp(-beta*z)/(1-exp(-beta)), r(z|m=1) = beta*exp(-beta*(1-z))/(1-exp(-beta)),
    the DVAE++ exponential-smoothing components (see sample_from_qz_given_x). logit_weight=0
    (q=0.5) gives the true prior p(z); logit_weight=m_logits gives the true q(z|x) -- unlike
    the Normal(qzm, qzv) moment-matched proxy used for VAEMixin.get_latent_representation
    compatibility, which is not the density z is actually sampled from. Not summed over the
    last dim -- callers sum, matching torch.distributions.Normal.log_prob's convention.
    """
    log_norm = torch.log(beta) - torch.log1p(-torch.exp(-beta))
    log_r0 = log_norm - beta * z
    log_r1 = log_norm - beta * (1 - z)
    log_q = logsigmoid(logit_weight)
    log_1mq = logsigmoid(-logit_weight)
    return logsumexp(torch.stack([log_q + log_r1, log_1mq + log_r0], dim=-1), dim=-1)


class _SmoothingPosterior:
    """Duck-typed q(z|x) for CodedVAESmoothing/CodedVAEECC: exposes .loc/.scale (the
    moment-matched Gaussian mean/std, all VAEMixin.get_latent_representation reads) and
    .log_prob (the *true*, non-Gaussian DVAE++ mixture density) -- a real
    torch.distributions.Normal(loc, scale) would answer .log_prob(z) under the wrong
    distribution family entirely, silently biasing any exact-likelihood consumer
    (e.g. marginal_ll).
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, logit_weight: torch.Tensor, beta: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self._logit_weight = logit_weight
        self._beta = beta

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return _log_smoothing_density(z, self._logit_weight, self._beta)


def _marginal_ll_with_prior(
    module: VAE,
    tensors: TensorDict,
    n_mc_samples: int,
    log_p_z: Callable[[torch.Tensor], torch.Tensor],
    return_mean: bool = False,
    n_mc_samples_per_pass: int = 1,
):
    """Same algorithm as scvi.module.VAE.marginal_ll, but with the prior log p(z) computed by
    log_p_z(z) (already summed over the latent dim) instead of the parent's hardcoded
    standard-normal assumption, which holds for none of this module's binary-latent priors.
    """
    batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
    to_sum = []
    if n_mc_samples_per_pass > n_mc_samples:
        n_mc_samples_per_pass = n_mc_samples
    n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
    for _ in range(n_passes):
        inference_outputs, _, losses = module.forward(tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass})
        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        reconst_loss = losses.dict_sum(losses.reconstruction_loss)

        p_z = log_p_z(z)
        p_x_zl = -reconst_loss
        q_z_x = qz.log_prob(z).sum(dim=-1)
        log_prob_sum = p_z + p_x_zl - q_z_x

        if not module.use_observed_lib_size:
            local_library_log_means, local_library_log_vars = module._compute_local_library_params(batch_index)
            p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
            q_l_x = ql.log_prob(library).sum(dim=-1)
            log_prob_sum += p_l - q_l_x

        to_sum.append(log_prob_sum)
    to_sum = torch.cat(to_sum, dim=0)
    batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
    if return_mean:
        batch_log_lkl = torch.mean(batch_log_lkl).item()
    else:
        batch_log_lkl = batch_log_lkl.cpu()
    return batch_log_lkl


class CodedVAE(VAE):
    """CodedVI module (not used in the paper): binary latent space via deterministic convex-combination reparameterization.

    c = m (no ECC yet). The bit posterior q(m|x) is produced by an encoder; per-bit Gaussian
    mixture component parameters (mu0, sigma0, mu1, sigma1) are learnable but do not depend on x.
    """

    def __init__(
        self,
        n_input: int,
        n_bits: int = 32,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(n_input, n_latent=n_bits, n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate, **kwargs)
        self.n_bits = n_bits
        # VAE does not store n_hidden as an attribute; keep our own copy for bit_encoder
        self.n_hidden = n_hidden
        cat_list = [self.n_batch] if self.encode_covariates else []
        self.bit_encoder = BinaryLatentEncoder(
            n_input, n_bits, cat_list, n_hidden=self.n_hidden, n_layers=n_layers, dropout_rate=dropout_rate
        )

        # learnable mixture-component parameters, fixed per bit position (not a function of x)
        self.comp_mu0 = nn.Parameter(torch.full((n_bits,), -1.0))
        self.comp_mu1 = nn.Parameter(torch.full((n_bits,), 1.0))
        self.comp_log_sigma0 = nn.Parameter(torch.zeros(n_bits))
        self.comp_log_sigma1 = nn.Parameter(torch.zeros(n_bits))

        # fixed prior, not optimized
        self.register_buffer("prior_mu0", torch.tensor(-1.0))
        self.register_buffer("prior_mu1", torch.tensor(1.0))
        self.register_buffer("prior_sigma0", torch.tensor(1.0))
        self.register_buffer("prior_sigma1", torch.tensor(1.0))
        self.register_buffer("prior_m_logits", torch.zeros(n_bits))  # Bernoulli(0.5)

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        out = super().inference(x, batch_index, cont_covs, cat_covs, n_samples)
        # reuse the library computed by the parent; only replace the z-related parts

        x_ = torch.log1p(x) if self.log_variational else x
        cat_list = (batch_index,) if self.encode_covariates else ()
        m_logits = self.bit_encoder(x_, *cat_list)
        q_i = torch.sigmoid(m_logits)

        sigma0 = self.comp_log_sigma0.exp()
        sigma1 = self.comp_log_sigma1.exp()
        # a leading n_samples dim on the noise (not on q_i, which is deterministic given x)
        # gives z that shape too, matching the parent's n_samples-expanded library tensor
        sample_shape = q_i.shape if n_samples == 1 else (n_samples, *q_i.shape)
        eps0 = torch.randn(sample_shape, device=q_i.device, dtype=q_i.dtype)
        eps1 = torch.randn(sample_shape, device=q_i.device, dtype=q_i.dtype)
        z0 = self.comp_mu0 + sigma0 * eps0
        z1 = self.comp_mu1 + sigma1 * eps1
        z = q_i * z1 + (1 - q_i) * z0

        qzm = q_i * self.comp_mu1 + (1 - q_i) * self.comp_mu0
        # convex-combination variance; NOT the full-expectation mixture-variance formula
        qzv = q_i**2 * sigma1**2 + (1 - q_i) ** 2 * sigma0**2
        # overwrite the parent's stale "qz" (from its unused z_encoder pass) so that
        # VAEMixin.get_latent_representation, which prefers "qz" over "qz_m"/"qz_v", picks
        # up our binary-latent posterior instead. z is an affine combination of two
        # independent Gaussians given q_i, so this Normal is q(z|x) *exactly* -- not an
        # approximation (unlike the moment-matched proxy CodedVAESmoothing/CodedVAEECC need).
        qz = Normal(qzm, qzv.sqrt())

        out.update({"z": z, "qz": qz, "qzm": qzm, "qzv": qzv, "m_logits": m_logits})
        return out

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        m_logits = inference_outputs["m_logits"]
        pi = torch.sigmoid(m_logits)
        sigma0 = self.comp_log_sigma0.exp()
        sigma1 = self.comp_log_sigma1.exp()

        q1 = Normal(self.comp_mu1, sigma1)
        q0 = Normal(self.comp_mu0, sigma0)
        p1 = Normal(self.prior_mu1, self.prior_sigma1)
        p0 = Normal(self.prior_mu0, self.prior_sigma0)
        kl_z = (pi * kl_divergence(q1, p1) + (1 - pi) * kl_divergence(q0, p0)).sum(-1)

        kl_m = kl_divergence(
            Bernoulli(logits=m_logits),
            Bernoulli(logits=self.prior_m_logits.expand_as(m_logits)),
        ).sum(-1)

        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.use_observed_lib_size:
            kl_l = torch.zeros_like(kl_z)
        else:
            kl_l = kl_divergence(inference_outputs["ql"], generative_outputs["pl"]).sum(dim=-1)

        recon = -generative_outputs["px"].log_prob(x).sum(-1)

        loss = torch.mean(recon + kl_weight * (kl_z + kl_m) + kl_l)
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon,
            kl_local={"kl_z": kl_z, "kl_m": kl_m, "kl_l": kl_l},
        )

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples, return_mean=False, n_mc_samples_per_pass=1):
        """Same as scvi.module.VAE.marginal_ll, but the prior p(z) is the true per-bit
        Gaussian mixture 0.5*N(prior_mu0,prior_sigma0) + 0.5*N(prior_mu1,prior_sigma1), not
        the parent's hardcoded standard-normal N(0,1) assumption (which this model never has).
        q(z|x)=qz.log_prob(z) needs no fix here -- see the comment in inference().
        """

        def log_p_z(z):
            log0 = Normal(self.prior_mu0, self.prior_sigma0).log_prob(z)
            log1 = Normal(self.prior_mu1, self.prior_sigma1).log_prob(z)
            log_mix = logsumexp(torch.stack([log0, log1], dim=-1), dim=-1) - np.log(2.0)
            return log_mix.sum(dim=-1)

        return _marginal_ll_with_prior(self, tensors, n_mc_samples, log_p_z, return_mean, n_mc_samples_per_pass)


class CodedVAESmoothing(VAE):
    """Uncoded module: DVAE++ / Coded-DVAE fixed exponential smoothing transformation, no code.

    Parallel to :class:`CodedVAE`; shares the same encoder backbone (BinaryLatentEncoder) and
    decoder (inherited, unmodified `generative()`). The only difference is how z is generated:
    q(z_i|x) and p(z_i) both mix the same parameter-free pair of truncated-exponential
    components r(z|m=0)/r(z|m=1) (only the mixture weight differs: q_i vs 0.5), so
    KL(q(z|m)||p(z|m)) is identically 0 and the only KL term is the per-bit Bernoulli KL on m.

    z is left in [0, 1]^K and fed to `generative()` unmodified (option A from the spec); z is
    *not* reparameterized via `comp_mu*`/`comp_sigma*` here since there is nothing to learn for
    the z-transform in this variant.

    beta is a fixed hyperparameter (not a nn.Parameter, not optimized), ported from
    mariamartinezgarcia/codedVAE (default 10, matching the reference repo's default).
    """

    def __init__(
        self,
        n_input: int,
        n_bits: int = 32,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        beta: float = 10.0,
        **kwargs,
    ):
        super().__init__(n_input, n_latent=n_bits, n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate, **kwargs)
        self.n_bits = n_bits
        # VAE does not store n_hidden as an attribute; keep our own copy for bit_encoder
        self.n_hidden = n_hidden
        cat_list = [self.n_batch] if self.encode_covariates else []
        self.bit_encoder = BinaryLatentEncoder(
            n_input, n_bits, cat_list, n_hidden=self.n_hidden, n_layers=n_layers, dropout_rate=dropout_rate
        )

        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("prior_m_logits", torch.zeros(n_bits))  # Bernoulli(0.5)

        # Closed-form mean/second-moment of r(z|m=0), the truncated-exponential component on
        # [0, 1] with rate beta: r(z|m=0) = beta*exp(-beta*z)/(1-exp(-beta)).
        # r(z|m=1) is its mirror image z -> 1-z, so it has mean (1 - mu0) and the same variance.
        # Used only to build a Normal "qz" proxy: VAEMixin.get_latent_representation requires
        # qz.loc/qz.scale (for give_mean=True and unconditionally for return_dist), even though
        # the true q(z|x) here is a truncated-exponential mixture, not Gaussian.
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
        m_logits = self.bit_encoder(x_, *cat_list)
        q_i = torch.sigmoid(m_logits)

        z = sample_from_qz_given_x(q_i, self.beta, n_samples=n_samples)  # (batch, n_bits, n_samples)
        z = z.squeeze(-1) if n_samples == 1 else z.permute(2, 0, 1)  # (batch, n_bits) or (n_samples, batch, n_bits)

        # convex-combination mean/variance of the fixed r(z|m) mixture (mirrors the
        # convention used in CodedVAE.inference, not the full-expectation mixture-variance
        # formula)
        qzm = self.mu0 * (1 - q_i) + (1 - self.mu0) * q_i
        qzv = (q_i**2 + (1 - q_i) ** 2) * self.var0
        # overwrite the parent's stale "qz" (see CodedVAE.inference for why this is needed).
        # z is *not* Gaussian here (a truncated-exponential mixture), so unlike CodedVAE this
        # is a moment-matched proxy, not the exact density -- _SmoothingPosterior.log_prob
        # uses the true density instead, while still exposing .loc/.scale for
        # get_latent_representation.
        qz = _SmoothingPosterior(qzm, qzv.sqrt(), m_logits, self.beta)

        out.update({"z": z, "qz": qz, "qzm": qzm, "qzv": qzv, "m_logits": m_logits})
        return out

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        m_logits = inference_outputs["m_logits"]

        # KL(q(z|m)||p(z|m)) = 0 since both share the same r(z|m) transform; the only KL
        # term left is the per-bit Bernoulli KL on m.
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
        smoothing-transform mixture at weight 0.5 (see _log_smoothing_density), not the
        parent's hardcoded standard-normal N(0,1) assumption (which does not hold on [0,1]).
        q(z|x) is likewise fixed via _SmoothingPosterior.log_prob, set up in inference().
        """

        def log_p_z(z):
            zero_logit = torch.zeros((), device=z.device, dtype=z.dtype)
            return _log_smoothing_density(z, zero_logit, self.beta).sum(dim=-1)

        return _marginal_ll_with_prior(self, tensors, n_mc_samples, log_p_z, return_mean, n_mc_samples_per_pass)
