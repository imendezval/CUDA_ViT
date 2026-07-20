from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads: int
    tokens: int
    head_dim: int

    @property
    def label(self) -> str:
        return f"B{self.batch}_H{self.heads}_T{self.tokens}_Dh{self.head_dim}"

    @property
    def attention_matmul_flops(self) -> int:
        return 2 * self.batch * self.heads * self.tokens * self.tokens * self.head_dim

    @property
    def supports_flashattention(self) -> bool:
        return self.head_dim == 64 and self.tokens % 32 == 0


ATTENTION_SHAPES = (
    AttentionShape(1, 1, 32, 64),
    AttentionShape(2, 3, 64, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(2, 3, 197, 64),
)


ATTENTION_OP_SHAPES = (
    AttentionShape(2, 2, 16, 32),
    AttentionShape(4, 3, 32, 64),
    AttentionShape(2, 3, 197, 64),
    AttentionShape(2, 6, 197, 64),
    AttentionShape(2, 4, 65, 48),
)
