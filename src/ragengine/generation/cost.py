"""Token cost estimation from `config/models.json`. Unknown prices give `None`, never a guess."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float | None
    output_per_mtok: float | None


class PriceTable:
    def __init__(self, prices: dict[str, Price]) -> None:
        self.prices = prices

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        p = Path(path)
        if not p.exists():
            return cls({})
        raw = json.loads(p.read_text())["models"]
        return cls(
            {
                name: Price(v.get("input_per_mtok"), v.get("output_per_mtok"))
                for name, v in raw.items()
            }
        )

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        price = self.prices.get(model)
        if price is None or price.input_per_mtok is None or price.output_per_mtok is None:
            return None
        usd = (
            prompt_tokens * price.input_per_mtok + completion_tokens * price.output_per_mtok
        ) / 1e6
        return round(usd, 8)
