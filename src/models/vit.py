import torch
from torch import nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        emb_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels, emb_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

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
        x = self.patch_embed(x)  # split image into patches
        x = x.flatten(2).transpose(1, 2)  # flatten and permute
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)  # add class token
        x = torch.cat((cls_tokens, x), dim=1)  # concat cls + patch embeddings
        x = self.pos_drop(x + self.pos_embed)  # add positional embeddings
        x = self.encoder(x)  # transformer encoder
        x = self.norm(x)  # layer norm
        return self.head(x[:, 0])  # classification head on CLS
