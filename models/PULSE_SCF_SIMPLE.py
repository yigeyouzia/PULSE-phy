"""PULSE + SOFTS + direct FACT control, without reference-conditioned deviation."""

from models.PULSE_SCF import PULSESCFModel


class PULSESCFSimpleModel(PULSESCFModel):
    scf_variant = "simple"


Model = PULSESCFSimpleModel
