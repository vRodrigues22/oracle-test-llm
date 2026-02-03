from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict


@dataclass
class OracleDecision:
    """Decisão do oráculo tradicional (baseline)."""
    verdict: str
    score: float
    reason: str
    telemetry: Dict[str, Any]


def _normalize(s: str) -> str:
    """Normaliza texto para reduzir variações triviais."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def regex_match(output_real: str, pattern: str) -> OracleDecision:
    """PASS se o regex casar; caso contrário FAIL."""
    ok = re.search(pattern, output_real or "", flags=re.IGNORECASE) is not None
    return OracleDecision(
        verdict="PASS" if ok else "FAIL",
        score=1.0 if ok else 0.0,
        reason="regex_matched" if ok else "regex_not_matched",
        telemetry={"mode": "regex", "pattern": pattern},
    )


def similarity_threshold(output_real: str, expected: str, threshold: float = 0.75) -> OracleDecision:
    """PASS se similaridade de string >= threshold; caso contrário FAIL."""
    a, b = _normalize(output_real), _normalize(expected)
    sim = SequenceMatcher(None, a, b).ratio()
    ok = sim >= threshold
    return OracleDecision(
        verdict="PASS" if ok else "FAIL",
        score=float(sim),
        reason=f"similarity={sim:.3f} thr={threshold:.2f}",
        telemetry={"mode": "similarity", "threshold": threshold},
    )


def evaluate(case: dict, mode: str = "regex") -> OracleDecision:
    """
    Avalia um caso via baseline tradicional.

    mode:
      - 'regex': usa regex genérica (bom para mensagens de erro/padrões)
      - 'similarity': compara output_real vs expected_claim por similaridade
    """
    output_real = case.get("output_real", "")
    expected = case.get("expected_claim", "")

    if mode == "similarity":
        return similarity_threshold(output_real, expected, 0.75)

    # Regex genérica para autenticação/credenciais (ajuste ao seu dataset)
    pattern = r"(usu[aá]rio|senha).*(incorret)|credencia(is|l).*(inv[aá]lid|invalid)|acesso negado|n[aã]o autorizado"
    return regex_match(output_real, pattern)
