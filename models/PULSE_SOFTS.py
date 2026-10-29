"""PULSE + SOFTS control, with the same settings as PULSE_SCF."""

from models.PULSE_SCF import PULSESCFModel


class PULSESOFTSModel(PULSESCFModel):
    scf_variant = "softs"


Model = PULSESOFTSModel
