"""Flow Matching causal avec tokens jambe D, jambe G et haut du corps.

Ce script reutilise strictement le pipeline de
``train_fm_causal_windows_christine_npz.py`` et remplace seulement son modele:

* 3 tokens cibles: jambe droite (6), jambe gauche (6), haut du corps (17);
* 4 blocs de memoire: F gauche, M gauche, F droite, M droite;
* cross-attention par TransformerDecoder;
* uniquement GRFM[t-W+1:t] pour predire q[t].

Il permet donc une comparaison directe avec la premiere architecture causale.
"""
import sys

import torch
import torch.nn as nn

import train_fm_causal_windows_christine_npz as base


class TokenizedCausalFlowModel(nn.Module):
    def __init__(self, max_window, dim=128, heads=4, layers=3):
        super().__init__()

        # Trois tokens correspondant aux trois groupes de la cible bruitee x_t.
        self.embed_right_leg = nn.Linear(6, dim)
        self.embed_left_leg = nn.Linear(6, dim)
        self.embed_upper_body = nn.Linear(17, dim)
        self.target_segment = nn.Parameter(torch.randn(3, dim) * 0.02)

        # Quatre types de tokens GRFM, chacun contenant W instants passes/courants.
        self.embed_force_left = nn.Linear(3, dim)
        self.embed_moment_left = nn.Linear(3, dim)
        self.embed_force_right = nn.Linear(3, dim)
        self.embed_moment_right = nn.Linear(3, dim)
        self.condition_segment = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.condition_position = nn.Parameter(
            torch.randn(1, max_window, dim) * 0.01
        )

        self.time = nn.Sequential(
            base.TimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)

        self.out_right_leg = nn.Linear(dim, 6)
        self.out_left_leg = nn.Linear(dim, 6)
        self.out_upper_body = nn.Linear(dim, 17)

    def forward(self, x_t, t, condition):
        """Retourne v_theta(x_t, t, GRFM_passe) de forme (B, 29)."""
        time_embedding = self.time(t).unsqueeze(1)

        target_tokens = torch.stack(
            (
                self.embed_right_leg(x_t[:, 0:6]),
                self.embed_left_leg(x_t[:, 6:12]),
                self.embed_upper_body(x_t[:, 12:29]),
            ),
            dim=1,
        )
        target_tokens = target_tokens + self.target_segment.unsqueeze(0)
        target_tokens = target_tokens + time_embedding

        # NPZ: [F_left(3), M_left(3), F_right(3), M_right(3)].
        condition_blocks = (
            self.embed_force_left(condition[:, :, 0:3]),
            self.embed_moment_left(condition[:, :, 3:6]),
            self.embed_force_right(condition[:, :, 6:9]),
            self.embed_moment_right(condition[:, :, 9:12]),
        )
        position = self.condition_position[:, :condition.shape[1], :]
        memory = torch.cat(
            [
                block + position + self.condition_segment[index]
                for index, block in enumerate(condition_blocks)
            ],
            dim=1,
        )
        # La memoire contient exclusivement les instants <= t. Aucun masque
        # supplementaire n'est requis pour respecter la causalite externe.
        decoded = self.decoder(tgt=target_tokens, memory=memory)

        return torch.cat(
            (
                self.out_right_leg(decoded[:, 0]),
                self.out_left_leg(decoded[:, 1]),
                self.out_upper_body(decoded[:, 2]),
            ),
            dim=1,
        )


if __name__ == "__main__":
    # train_one() recherche cette classe dans le module de base. Le reste du
    # pipeline demeure donc strictement identique entre les deux experiences.
    base.CausalFlowModel = TokenizedCausalFlowModel
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "results_fm_causal_windows_tokenized"])
    base.main()
