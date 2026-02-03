from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """Preço por 1k tokens (input/output) em USD (opcional)."""
    input_per_1k: float
    output_per_1k: float


# Opcional: preencha se quiser estimar custo.
PRICES_USD: dict[str, Price] = {}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Retorna custo estimado em USD; 0.0 se não houver tabela de preços."""
    p = PRICES_USD.get(model)
    if not p:
        return 0.0
    return (input_tokens / 1000.0) * p.input_per_1k + (output_tokens / 1000.0) * p.output_per_1k
