"""Train every binary variant for two epochs on scvi-tools' synthetic data and read the bits."""
import numpy as np
import pytest
import scvi

from codedmrvi import CodedMRVI, CodedMRVIECC
from codedvi import CODEDVIECC, CODEDVISmoothing

K, R = 10, 10


@pytest.fixture
def adata():
    return scvi.data.synthetic_iid(batch_size=200, n_genes=100, n_batches=2, n_labels=3)


def _check_bits(model, n_cells):
    q = model.get_bit_probabilities()
    assert q.shape == (n_cells, K)
    assert np.all((q >= 0) & (q <= 1))


def test_coded_scvi(adata):
    CODEDVIECC.setup_anndata(adata, batch_key="batch")
    model = CODEDVIECC(adata, n_info_bits=K, rep_factor=R)
    model.train(max_epochs=2)
    _check_bits(model, adata.n_obs)
    assert model.get_hard_codewords().shape == (adata.n_obs, K * R)
    assert model.get_latent_representation().shape == (adata.n_obs, K * R)


def test_uncoded_scvi(adata):
    CODEDVISmoothing.setup_anndata(adata, batch_key="batch")
    model = CODEDVISmoothing(adata, n_bits=K)
    model.train(max_epochs=2)
    _check_bits(model, adata.n_obs)


def test_coded_mrvi(adata):
    CodedMRVIECC.setup_anndata(adata, sample_key="batch")
    model = CodedMRVIECC(adata, n_info_bits=K, rep_factor=R)
    model.train(max_epochs=2)
    _check_bits(model, adata.n_obs)
    assert model.get_latent_representation(give_z=False).shape == (adata.n_obs, K * R)


def test_uncoded_mrvi(adata):
    CodedMRVI.setup_anndata(adata, sample_key="batch")
    model = CodedMRVI(adata, n_latent_u=K)
    model.train(max_epochs=2)
    _check_bits(model, adata.n_obs)
