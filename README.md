# coded-scvi: binary, repetition-coded latents for scVI and MrVI

This package replaces the innermost Gaussian latent of two [scvi-tools](https://scvi-tools.org) models,
**scVI** and **MrVI**, with **K binary information bits protected by a repetition code**. Only the
prior and posterior of that one latent change. The encoder trunk, the decoder and, in MrVI, the
sample-aware pathway are the base model's own.

```
counts x_n ──► encoder ──► K bits (+ repetition code) ──► smoothed codeword ζ_n ──► decoder ──► x̂_n
                           └────────── the only change: replaces the Gaussian z_n / u_n ──────────┘
```

- Each of the K information bits is repeated r times, giving a codeword of length N_b = K·r. Each
  codeword bit is relaxed to [0, 1] with the DVAE++ smoothing transform, and the decoder reads this
  smoothed codeword.
- The encoder's per-coordinate evidence is combined by soft decoding into one probability per
  information bit, q̂_nk = P(bit k = 1 | x_n).
- The KL term is a sum of K closed-form Bernoulli KLs against Bernoulli(½).
- A cell is represented by **K probabilities**, many close to 0 or 1, that can be read directly as
  on/off calls. A Gaussian latent is identifiable only up to rotation; a factorized Bernoulli prior
  allows only permutations and flips of the bits, so bits of independently trained models can be
  matched.

Paper: *Binary Latents for Single-Cell Generative Models: Interpretable Cell-State Codes in
scvi-tools*, X. Shen, M. Martínez-García, P. M. Olmos (2026). The scripts that reproduce the paper's
experiments will be added to this repository.

## Models

| Class | Base model | Latent | Main arguments |
|---|---|---|---|
| `codedvi.CODEDVIECC` | scVI | **coded**: K bits, repetition code of rate 1/r | `n_info_bits` (K), `rep_factor` (r), `beta` |
| `codedvi.CODEDVISmoothing` | scVI | uncoded: K bits, no code (r = 1) | `n_bits` (K), `beta` |
| `codedmrvi.CodedMRVIECC` | MrVI (`scvi.external.mrvi_torch.TorchMRVI`) | **coded** cell-state latent u | `n_info_bits`, `rep_factor`, `beta` |
| `codedmrvi.CodedMRVI` | MrVI | uncoded cell-state latent u | `n_latent_u` (K), `beta` |

`beta` is the fixed temperature of the smoothing transform (default 10). All other arguments
(`n_hidden`, `n_layers`, `gene_likelihood`, …, and MrVI's model kwargs) are passed through as in
the base model, and `setup_anndata` takes the same keys as the base model.

## Installation

Requires Python ≥ 3.10 and scvi-tools 1.4 (tested with scvi-tools 1.4.2, PyTorch 2.14).

```bash
git clone https://github.com/xshen-jpg/coded-scvi.git
cd coded-scvi
pip install -e .
```

## Quickstart

### Coded scVI

```python
import scanpy as sc
import scvi
from codedvi import CODEDVIECC

adata = scvi.data.pbmc_dataset()                 # any AnnData with raw counts
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

CODEDVIECC.setup_anndata(adata, layer="counts", batch_key="batch")
model = CODEDVIECC(adata, n_info_bits=10, rep_factor=10)   # 10 bits, codeword length 100
model.train()

q = model.get_bit_probabilities()             # (n_cells, K): P(bit k = 1 | x), the readable code
bits = model.get_hard_information_bits()      # (n_cells, K): 1[q > 0.5]
zeta = model.get_latent_representation()      # (n_cells, K*r): smoothed codeword, the decoder input
```

### Coded MrVI

```python
from codedmrvi import CodedMRVIECC

# adata as above, with a column of sample (donor) ids in adata.obs["sample"]
CodedMRVIECC.setup_anndata(adata, layer="counts", sample_key="sample")
model = CodedMRVIECC(adata, n_info_bits=10, rep_factor=10)
model.train()

q = model.get_bit_probabilities()                              # bits of the cell-state latent u
u = model.get_latent_representation(give_z=False)              # smoothed codeword of u
```

### Reading the bits

Averaging `q` over the cells of each annotated cell type gives a bits × cell-types matrix, in which
entries near 0 or 1 mark a bit that is off or on for that type:

```python
import pandas as pd

labels = adata.obs["str_labels"].values      # the cell-type column of your data
activation = pd.DataFrame(q).groupby(labels).mean().T   # bits x cell types
```

## Settings used in the paper

- K = 10 information bits, `beta = 10`; repetition rate 1/10 or 1/15 (`rep_factor=10` or `15`).
  The vanilla baseline is the base model with a 10-dimensional Gaussian latent.
- 2,000 highly variable genes; raw counts in `layer="counts"`.
- Training: at most 200 epochs, KL warm-up over the first 20 epochs, and early stopping on the
  validation loss with patience 30, counted after the warm-up:

```python
model.train(max_epochs=200, early_stopping=True, early_stopping_monitor="validation_loss",
            early_stopping_patience=30, early_stopping_warmup_epochs=20,
            plan_kwargs={"n_epochs_kl_warmup": 20})
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests train every variant for two epochs on scvi-tools' synthetic data and check the shape and
range of the bit probabilities.

## Citation

A citation will be added when the paper is published.

## License

BSD 3-Clause; see [LICENSE](LICENSE). The models build on
[scvi-tools](https://github.com/scverse/scvi-tools) (BSD 3-Clause).
