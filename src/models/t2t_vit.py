import torch
import torch.nn.functional as F
from torch import nn


class T2TModule(nn.Module):
    """
    Aggregates local image patches into tokens.
    """

    def __init__(self, in_channels: int = 3, emb_dim: int = 768):
        super().__init__()
        # first soft split: 7x7 conv, stride 4
        self.soft_split1 = nn.Conv2d(
            in_channels, emb_dim // 2, kernel_size=7, stride=4, padding=2
        )
        # second soft split: 3x3 conv, stride 2
        self.soft_split2 = nn.Conv2d(
            emb_dim // 2, emb_dim, kernel_size=3, stride=2, padding=1
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.soft_split1(x)  # (B, emb_dim//2, H1, W1)
        x = F.gelu(x)
        x = self.soft_split2(x)  # (B, emb_dim, H2, W2)
        x = F.gelu(x)
        # flatten to tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N_tokens, emb_dim)
        x = self.norm(x)
        return x


class T2TViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 100,
        emb_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.t2t = T2TModule(in_channels, emb_dim)

        # compute number of tokens after T2T
        num_patches = (img_size // 8) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x[:, 0])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t2t(x)  # apply T2T module
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        return x
