"""SCF settings, independent of PULSE's Router dimensions and CLI defaults."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SCFConfig:
    d_model: int = 128
    d_core: int = 32
    d_ff: int = 256
    e_layers: int = 1
    dropout: float = 0.1
    fact_dilation: int = 2
    num_kernels: int = 2
    gate_hidden: int = 32
    gate_init: float = 0.1

    @classmethod
    def from_configs(cls, configs):
        # Only explicit scf_* fields override SCF settings. In particular,
        # configs.d_model continues to control the inherited PULSE Router.
        values = {
            field.name: getattr(configs, "scf_" + field.name, field.default)
            for field in fields(cls)
        }
        settings = cls(**values)
        for name in ("d_model", "d_core", "d_ff", "e_layers", "fact_dilation",
                     "num_kernels", "gate_hidden"):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("scf_{} must be a positive integer".format(name))
        if not 0 <= settings.dropout < 1:
            raise ValueError("scf_dropout must be in [0, 1)")
        if not 0 < settings.gate_init < 1:
            raise ValueError("scf_gate_init must be in (0, 1)")
        return settings

    def as_dict(self):
        return asdict(self)
