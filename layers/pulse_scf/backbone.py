"""SOFTS-shaped backbone with optional model-specific FACT interaction."""

from torch import nn

from .interaction import SCFInteraction, SCFSTAR
from .vendor.softs_embedding import DataEmbedding_inverted
from .vendor.softs_encoder import Encoder, EncoderLayer


class SCFBackbone(nn.Module):
    def __init__(self, seq_len, pred_len, channels, settings, variant="full"):
        super().__init__()
        if variant not in {"softs", "simple", "full"}:
            raise ValueError("SCF backbone variant must be softs, simple or full")
        self.embedding = DataEmbedding_inverted(seq_len, settings.d_model, settings.dropout)
        layers = []
        for _ in range(settings.e_layers):
            interaction = (SCFSTAR(settings.d_model, settings.d_core) if variant == "softs"
                           else SCFInteraction(channels, seq_len, settings, variant))
            layers.append(EncoderLayer(
                interaction, settings.d_model, settings.d_ff,
                dropout=settings.dropout, activation="gelu",
            ))
        self.encoder = Encoder(layers)
        self.projection = nn.Linear(settings.d_model, pred_len)

    def forward(self, x_in, scf_state=None):
        # PULSE already encodes timestamps into its reference. Only the C actual
        # variables enter STAR and the FACT grid; there are no extra tokens.
        z = self.embedding(x_in, None)
        z, _ = self.encoder(z, scf_state=scf_state)
        return self.projection(z).permute(0, 2, 1)
