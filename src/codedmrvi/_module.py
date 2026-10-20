"""Uncoded/coded binary-latent variants of MrVI's sample-unaware latent `u`.

MrVI (:class:`~scvi.external.mrvi_torch.TorchMRVAE`) factorizes its latent representation into
two parts: `u`, a sample-*unaware* "cell state" bottleneck (the direct structural analogue of
plain scVI's single latent `z`), and `z = z_base(u, sample) + eps`, a sample-*aware* latent
obtained by passing `u` through an attention-based encoder conditioned on the sample/donor
identity. Only the decoder (`p(x|z)`) and the `u -> z` attention encoder ever consume `u`'s
*value*; both treat it as an opaque continuous vector. That makes `u` the correct injection
point for the same binary/coded-latent recipe used for scVI's `z` in the `codedvi` package:
swap the Gaussian `EncoderXU` for a bit-logit head, reparameterize each bit through the
DVAE++ fixed exponential-smoothing transform (ported unchanged from
mariamartinezgarcia/codedVAE), and leave `EncoderUZ`/`DecoderZXAttention`/the `z`-side KL term
completely untouched.

The DVAE++ math (`dclamp`, `sample_from_qz_given_x`, `_log_smoothing_density`,
`_SmoothingPosterior`) and the repetition-code helper (`_build_repetition_code`) are ported
verbatim from `codedvi/_module.py` and `codedvi/_module_ecc.py` -- this file is intentionally
self-contained (no import from the `codedvi` package) since it targets a different scvi-tools
major version / Python version, installed into a separate virtualenv.
"""

from typing import Dict

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.init as init
from scvi import REGISTRY_KEYS
from scvi.external.mrvi_torch._components import ConditionalNormalization
from scvi.external.mrvi_torch._module import TorchMRVAE
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import Bernoulli, kl_divergence
from torch.nn.functional import logsigmoid

TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# DVAE++ exponential-smoothing transform (ported from codedvi/_module.py, which itself ports
# mariamartinezgarcia/codedVAE line-for-line; see that file for the reference-repo mapping).
# ---------------------------------------------------------------------------


class _DifferentiableClamp(torch.autograd.Function):
    """Clamp with gradient 1 everywhere (identity backward), avoiding dead gradients at the bounds."""

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

    Parameters
    ----------
    qi
        q(bit=1|x), shape (batch, n_bits).
    beta
        Fixed temperature controlling the decay of the smoothing exponentials.
    n_samples
        Number of samples to draw per bit.

    Returns
    -------
    z of shape (batch, n_bits, n_samples).
    """
    epsilon = 1e-6

    qi = qi.unsqueeze(2).repeat(1, 1, n_samples)
    ones = torch.ones_like(qi)

    qi = dclamp(qi, 0, 1 - 1e-3)
    rho = torch.rand_like(qi)

    b = (rho + torch.exp(-beta) * (qi - rho)) / (ones - qi) - ones
    c = -(qi * torch.exp(-beta)) / (ones - qi)

    dif = torch.sqrt(torch.pow(b, 2) - 4 * c) - b
    dif = torch.where(dif <= 0, torch.full_like(dif, epsilon), dif)

    z = (-1 / beta) * torch.log(dif / 2)
    return z


def _log_smoothing_density(z: torch.Tensor, logit_weight: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Per-element log[q*r(z|m=1) + (1-q)*r(z|m=0)], q = sigmoid(logit_weight). Not summed."""
    log_norm = torch.log(beta) - torch.log1p(-torch.exp(-beta))
    log_r0 = log_norm - beta * z
    log_r1 = log_norm - beta * (1 - z)
    log_q = logsigmoid(logit_weight)
    log_1mq = logsigmoid(-logit_weight)
    return torch.logsumexp(torch.stack([log_q + log_r1, log_1mq + log_r0], dim=-1), dim=-1)


class _SmoothingPosterior:
    """Duck-typed q(u|x): exposes .loc/.scale (moment-matched Gaussian, for any consumer that
    expects those two attributes) and .log_prob (the true, non-Gaussian DVAE++ density).
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, logit_weight: torch.Tensor, beta: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.mean = loc
        self._logit_weight = logit_weight
        self._beta = beta

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return _log_smoothing_density(z, self._logit_weight, self._beta)


def _build_repetition_code(n_info_bits: int, rep_factor: int) -> torch.Tensor:
    """Rate 1/rep_factor repetition code generator matrix G, shape (n_info_bits, n_bits)."""
    n_bits = n_info_bits * rep_factor
    G = torch.zeros(n_info_bits, n_bits)
    for i in range(n_info_bits):
        G[i, i * rep_factor : (i + 1) * rep_factor] = 1.0
    return G


def _smoothing_moments(beta: torch.Tensor):
    """Closed-form mean/variance of r(z|m=0), the truncated-exponential component of the DVAE++
    smoothing transform on [0, 1] with rate beta. r(z|m=1) is its mirror image z -> 1-z.
    """
    mu0 = 1.0 / beta - 1.0 / (torch.exp(beta) - 1.0)
    m2_0 = (
        -torch.exp(-beta) - 2 * torch.exp(-beta) / beta + 2 * (1 - torch.exp(-beta)) / beta**2
    ) / (1 - torch.exp(-beta))
    return mu0, m2_0 - mu0**2


# ---------------------------------------------------------------------------
# Binary encoder x -> bit logits, architecturally parallel to MrVI's EncoderXU (same
# sample-conditioned backbone: fc1/conditional_norm1/fc2/conditional_norm2/sample_embed) but
# ending in a plain bit-logit head instead of a NormalDistOutputNN.
# ---------------------------------------------------------------------------


class BinaryEncoderXU(nn.Module):
    """Encoder from ``x`` to bit logits, sample-conditioned like MrVI's EncoderXU."""

    def __init__(
        self,
        n_input: int,
        n_bits: int,
        n_sample: int,
        n_hidden: int = 128,
        activation=nn.functional.gelu,
    ):
        super().__init__()
        self.activation = activation

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.conditional_norm1 = ConditionalNormalization(n_hidden, n_sample)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.conditional_norm2 = ConditionalNormalization(n_hidden, n_sample)

        self.sample_embed = nn.Embedding(n_sample, n_hidden)
        init.normal_(self.sample_embed.weight, std=0.1)

        self.m_logits = nn.Linear(n_hidden, n_bits)

    @auto_move_data
    def forward(self, x: torch.Tensor, sample_covariate: torch.Tensor, training: bool = None) -> torch.Tensor:
        x_feat = torch.log1p(x)
        x_feat = self.fc1(x_feat)
        x_feat = self.conditional_norm1(x_feat, sample_covariate, training=training)
        x_feat = self.activation(x_feat)
        x_feat = self.fc2(x_feat)
        x_feat = self.conditional_norm2(x_feat, sample_covariate, training=training)
        x_feat = self.activation(x_feat)
        sample_effect = self.sample_embed(sample_covariate.squeeze(-1).to(torch.int64))
        return self.m_logits(x_feat + sample_effect)


def _coerce_inference_inputs(x, sample_index):
    """Same numpy->Tensor coercion TorchMRVAE.inference does at its top (needed because some
    scvi-tools MrVI callers, e.g. get_aggregated_posterior, invoke module.inference directly on
    an ndarray dataloader's raw batch instead of routing through module.forward).
    """
    if type(x).__name__ != "Tensor" and x is not None:
        x = torch.Tensor(x)
    if type(sample_index).__name__ != "Tensor" and sample_index is not None:
        sample_index = torch.Tensor(sample_index)
    return x, sample_index


def _finish_inference(module: TorchMRVAE, x, u, qu, m_logits, sample_index_cf) -> TensorDict:
    """Shared tail of inference() from the point `u` has been produced: unchanged from
    TorchMRVAE.inference (EncoderUZ, the eps posterior/sample, and the library statistic).
    """
    z_base, eps = module.qz(u, sample_index_cf)
    qeps_ = eps
    qeps = None
    if qeps_.shape[-1] == 2 * module.n_latent:
        loc_, scale_ = qeps_[..., : module.n_latent], qeps_[..., module.n_latent :]
        qeps = dist.Normal(loc_, nn.functional.softplus(scale_) + 1e-3)
        noise = torch.normal(mean=0.0, std=1.0, size=qeps.mean.shape, device=qeps.mean.device, dtype=qeps.mean.dtype)
        eps = qeps.mean + noise * qeps.scale
    z = z_base + eps
    library = torch.log(x.sum(1, keepdims=True))
    return {"qu": qu, "qeps": qeps, "eps": eps, "u": u, "z": z, "z_base": z_base, "library": library, "m_logits": m_logits}


def _kl_z_term(module: TorchMRVAE, inference_outputs: TensorDict) -> torch.Tensor:
    """KL for the sample-aware residual eps = z - z_base.

    Device note: ``pz_scale`` is an ``nn.Parameter`` only when ``learn_z_u_prior_scale`` is
    True; otherwise it is the plain ``z_u_prior_scale`` tensor, which never moves with the
    module and so stays on the CPU under CUDA. Upstream computes this term on the CPU and
    then moves the result back; we instead bring the scale to the data, which keeps the whole
    term on the accelerator and avoids a per-step round trip. Returning a CPU tensor here is
    what made ``kl_u + kl_z`` raise a device mismatch on GPU while passing on CPU-only runs.
    """
    if not module.z_u_prior:
        return 0.0
    eps = inference_outputs["z"] - inference_outputs["z_base"]
    peps = dist.Normal(0, torch.exp(module.pz_scale.to(eps.device)))
    return (-peps.log_prob(eps).sum(-1))


@auto_move_data
def _generative_px_h(module: TorchMRVAE, z, library, batch_index, label_index, transform_batch=None) -> TensorDict:
    """Same px/h computation as TorchMRVAE.generative, but skips its `pu` branch entirely (not
    just the unused `u_prior_mixture=True` label-mixture path, but also the `u_prior_mixture=
    False` branch, which does `torch.exp(self.u_prior_scale)` on `self.u_prior_scale` still
    being a plain Python float rather than a tensor in scvi-tools 1.4.2 -- a real upstream
    TypeError, sidestepped here since `pu` is never read by this module's `loss()` in the first
    place (the u-side KL is the closed-form Bernoulli KL on `m`, computed directly).
    """
    if transform_batch is not None:
        batch_index = torch.ones_like(batch_index) * transform_batch
    library_exp = torch.exp(library)
    px = module.px(z, batch_index, size_factor=library_exp)
    h = px.mean / library_exp
    return {"px": px, "pu": None, "h": h}


class CodedMRVAESmoothing(TorchMRVAE):
    """Uncoded variant: MrVI with `u` produced by the DVAE++ smoothing transform instead of a
    Gaussian. Each dimension of `u` is one independently-encoded bit (no error-correcting code).

    KL(q(u_i|m_i) || p(u_i|m_i)) is identically 0, since q and p share the same fixed r(u|m)
    transform and differ only in mixture weight (q_i vs. 0.5) -- exactly as in
    codedvi.CodedVAESmoothing. The only KL term for `u` is therefore the per-bit Bernoulli KL on
    `m`, computed in closed form (not by Monte-Carlo log-prob difference).
    """

    def __init__(
        self,
        n_input: int,
        n_sample: int,
        n_batch: int,
        n_labels: int,
        n_latent: int = 30,
        n_latent_u: int = 10,
        encoder_n_hidden: int = 128,
        encoder_n_layers: int = 2,
        beta: float = 10.0,
        **kwargs,
    ):
        kwargs.pop("u_prior_mixture", None)
        super().__init__(
            n_input=n_input,
            n_sample=n_sample,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent=n_latent,
            n_latent_u=n_latent_u,
            encoder_n_hidden=encoder_n_hidden,
            encoder_n_layers=encoder_n_layers,
            u_prior_mixture=False,
            **kwargs,
        )
        self.n_bits = self.n_latent_u
        self.qu = BinaryEncoderXU(
            n_input=self.n_input, n_bits=self.n_bits, n_sample=self.n_sample, n_hidden=self.encoder_n_hidden
        )
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("prior_m_logits", torch.zeros(self.n_bits))
        mu0, var0 = _smoothing_moments(self.beta)
        self.register_buffer("mu0", mu0)
        self.register_buffer("var0", var0)

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        sample_index: torch.Tensor,
        mc_samples: int = None,
        cf_sample: torch.Tensor = None,
        use_mean: bool = False,
    ) -> TensorDict:
        x, sample_index = _coerce_inference_inputs(x, sample_index)
        m_logits = self.qu(x, sample_index, training=self.training)
        q_i = torch.sigmoid(m_logits)

        qzm = self.mu0 * (1 - q_i) + (1 - self.mu0) * q_i
        qzv = (q_i**2 + (1 - q_i) ** 2) * self.var0
        qu = _SmoothingPosterior(qzm, qzv.sqrt(), m_logits, self.beta)

        if use_mean:
            u = qzm
        else:
            n_samples = mc_samples if mc_samples is not None else 1
            u_raw = sample_from_qz_given_x(q_i, self.beta, n_samples=n_samples)  # (batch, n_bits, n_samples)
            u = u_raw.squeeze(-1) if mc_samples is None else u_raw.permute(2, 0, 1)

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        return _finish_inference(self, x, u, qu, m_logits, sample_index_cf)

    def generative(self, z, library, batch_index, label_index, transform_batch=None) -> TensorDict:
        return _generative_px_h(self, z, library, batch_index, label_index, transform_batch)

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0) -> LossOutput:
        m_logits = inference_outputs["m_logits"]
        kl_u = kl_divergence(
            Bernoulli(logits=m_logits), Bernoulli(logits=self.prior_m_logits.expand_as(m_logits))
        ).sum(-1)
        kl_z = _kl_z_term(self, inference_outputs)

        reconstruction_loss = -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        kl_local = kl_u + kl_z
        loss = reconstruction_loss + kl_weight * kl_local

        if self.scale_observations:
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].flatten().to(torch.int64)
            prefactors = self.n_obs_per_sample.to(self.device)[sample_index]
            loss = loss / prefactors

        loss = torch.mean(loss)
        return LossOutput(loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local)


class CodedMRVAEECC(TorchMRVAE):
    """Coded variant: layers a rate-1/rep_factor repetition-code ECC on top of
    CodedMRVAESmoothing's per-bit smoothing transform, exactly mirroring codedvi.CodedVAEECC.

    `n_info_bits` information bits are each repeated `rep_factor` times into a length
    `n_latent_u = n_info_bits * rep_factor` codeword; `BinaryEncoderXU` predicts
    q(codeword_bit=1|x) independently per position, a soft-decoding step (naive-Bayes
    combination in log-space) recovers the (less noisy) info-bit posterior, which is
    re-broadcast to all codeword positions before the DVAE++ smoothing reparameterization.
    """

    def __init__(
        self,
        n_input: int,
        n_sample: int,
        n_batch: int,
        n_labels: int,
        n_latent: int = 30,
        n_info_bits: int = 10,
        rep_factor: int = 4,
        encoder_n_hidden: int = 128,
        encoder_n_layers: int = 2,
        beta: float = 10.0,
        **kwargs,
    ):
        n_latent_u = n_info_bits * rep_factor
        kwargs.pop("u_prior_mixture", None)
        super().__init__(
            n_input=n_input,
            n_sample=n_sample,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent=n_latent,
            n_latent_u=n_latent_u,
            encoder_n_hidden=encoder_n_hidden,
            encoder_n_layers=encoder_n_layers,
            u_prior_mixture=False,
            **kwargs,
        )
        self.n_info_bits = n_info_bits
        self.rep_factor = rep_factor
        self.n_bits = n_latent_u
        self.qu = BinaryEncoderXU(
            n_input=self.n_input, n_bits=self.n_bits, n_sample=self.n_sample, n_hidden=self.encoder_n_hidden
        )
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("prior_m_logits", torch.zeros(n_info_bits))
        G = _build_repetition_code(n_info_bits, rep_factor)
        self.register_buffer("G", G)
        self.register_buffer("H", G.t())
        mu0, var0 = _smoothing_moments(self.beta)
        self.register_buffer("mu0", mu0)
        self.register_buffer("var0", var0)

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        sample_index: torch.Tensor,
        mc_samples: int = None,
        cf_sample: torch.Tensor = None,
        use_mean: bool = False,
    ) -> TensorDict:
        x, sample_index = _coerce_inference_inputs(x, sample_index)
        c_logits = self.qu(x, sample_index, training=self.training)  # (batch, n_bits) codeword-position logits
        c_probs = torch.sigmoid(c_logits)
        c_probs = dclamp(c_probs, 1e-6, 1 - 1e-6)

        logpm1 = torch.log(c_probs) @ self.H
        logpm0 = torch.log(1 - c_probs) @ self.H
        log_marginals = torch.stack([logpm0, logpm1], dim=-1)
        log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
        m_logits = log_marginals_norm[..., 1] - log_marginals_norm[..., 0]  # (batch, n_info_bits)
        m_probs = log_marginals_norm[..., 1].exp()

        qc = m_probs @ self.G  # (batch, n_bits)

        qzm = self.mu0 * (1 - qc) + (1 - self.mu0) * qc
        qzv = (qc**2 + (1 - qc) ** 2) * self.var0
        qu = _SmoothingPosterior(qzm, qzv.sqrt(), m_logits @ self.G, self.beta)

        if use_mean:
            u = qzm
        else:
            n_samples = mc_samples if mc_samples is not None else 1
            u_raw = sample_from_qz_given_x(qc, self.beta, n_samples=n_samples)
            u = u_raw.squeeze(-1) if mc_samples is None else u_raw.permute(2, 0, 1)

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        outputs = _finish_inference(self, x, u, qu, m_logits, sample_index_cf)
        # ``m_logits`` are the decoded K information-bit logits.  Preserve the raw
        # N-position codeword logits for ECC recovery/robustness diagnostics.
        outputs["c_logits"] = c_logits
        return outputs

    def generative(self, z, library, batch_index, label_index, transform_batch=None) -> TensorDict:
        return _generative_px_h(self, z, library, batch_index, label_index, transform_batch)

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0) -> LossOutput:
        m_logits = inference_outputs["m_logits"]
        kl_u = kl_divergence(
            Bernoulli(logits=m_logits), Bernoulli(logits=self.prior_m_logits.expand_as(m_logits))
        ).sum(-1)
        kl_z = _kl_z_term(self, inference_outputs)

        reconstruction_loss = -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        kl_local = kl_u + kl_z
        loss = reconstruction_loss + kl_weight * kl_local

        if self.scale_observations:
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].flatten().to(torch.int64)
            prefactors = self.n_obs_per_sample.to(self.device)[sample_index]
            loss = loss / prefactors

        loss = torch.mean(loss)
        return LossOutput(loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local)
