"""Binary-latent scVI: uncoded (CODEDVISmoothing) and repetition-coded (CODEDVIECC) variants."""

from ._model import CODEDVI, CODEDVIECC, CODEDVISmoothing
from ._module import BinaryLatentEncoder, CodedVAE, CodedVAESmoothing
from ._module_ecc import CodedVAEECC

__all__ = [
    "CODEDVIECC",
    "CODEDVISmoothing",
    "CodedVAEECC",
    "CodedVAESmoothing",
    "CODEDVI",
    "CodedVAE",
    "BinaryLatentEncoder",
]
