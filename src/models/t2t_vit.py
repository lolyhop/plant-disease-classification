import torch
import torch.nn.functional as F
from torch import nn


class T2TModule(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 768, t2t_dim: int = 64):
        super().__init__()
        self.soft_split0 = nn.Unfold(kernel_size=7, stride=4, padding=2)
        self.project0 = nn.Linear(7 * 7 * in_channels, t2t_dim)
        self.attn1 = nn.TransformerEncoderLayer(
            d_model=t2t_dim,
            nhead=1,
            dim_feedforward=t2t_dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.soft_split1 = nn.Unfold(kernel_size=3, stride=2, padding=1)
        self.project1 = nn.Linear(3 * 3 * t2t_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.soft_split0(x).transpose(1, 2)
        x = self.project0(x)
        x = self.attn1(x)
        h1 = (h + 2 * 2 - 7) // 4 + 1
        w1 = (w + 2 * 2 - 7) // 4 + 1
        x = x.transpose(1, 2).reshape(b, -1, h1, w1)
        x = self.soft_split1(x).transpose(1, 2)
        x = self.project1(x)
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
        x = self.t2t(x)
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        return x
