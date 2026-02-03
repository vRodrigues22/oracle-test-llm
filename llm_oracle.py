from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from jsonschema import validate as js_validate

from cache import SQLiteCache
from config import settings
from costs import estimate_cost_usd
from schemas import ORACLE_OUTPUT_SCHEMA


@dataclass
class OracleDecision:
    """Decisão do oráculo LLM."""
    verdict: str
    score: float
    reason: str
    evidence: List[str]
    missing_requirements: List[str]
    telemetry: Dict[str, Any]


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _safe_parse_json(text: str) -> Optional[dict]:
    """Tenta parsear JSON mesmo quando o modelo coloca lixo ao redor."""
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i : j + 1])
        except Exception:
            return None
    return None


def _validate_schema(obj: dict) -> None:
    js_validate(instance=obj, schema=ORACLE_OUTPUT_SCHEMA)


class BaseProvider:
    name: str = "base"

    def generate(self, system: str, user_input: str, seed: Optional[int]) -> Tuple[str, dict]:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY não configurada no .env.")
        from openai import OpenAI

        # O SDK também aceita OPENAI_API_KEY via ambiente; aqui mantemos explícito.
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, system: str, user_input: str, seed: Optional[int]) -> Tuple[str, dict]:
        # Nota: removi 'seed' do request por compatibilidade de assinatura/stubs.
        resp = self.client.responses.create(
            model=settings.llm_model,
            instructions=system,
            input=user_input,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_output_tokens=settings.max_output_tokens,
        )

        text = getattr(resp, "output_text", "") or ""

        usage = getattr(resp, "usage", None)
        usage_dict = {}
        if usage:
            usage_dict = {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }

        return text, usage_dict


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não configurada no .env.")
        from google import genai

        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, system: str, user_input: str, seed: Optional[int]) -> Tuple[str, dict]:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=system,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_output_tokens=settings.max_output_tokens,
        )
        resp = self.client.models.generate_content(
            model=settings.llm_model,
            contents=user_input,
            config=cfg,
        )
        text = getattr(resp, "text", "") or ""

        usage_dict: dict = {}
        u = getattr(resp, "usage_metadata", None)
        if u:
            usage_dict = {
                "input_tokens": int(getattr(u, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(u, "candidates_token_count", 0) or 0),
                "total_tokens": int(getattr(u, "total_token_count", 0) or 0),
            }
        return text, usage_dict


class LlamaProvider(BaseProvider):
    """Endpoint OpenAI-compatible (ex.: Ollama /v1)."""
    name = "llama"

    def __init__(self):
        if not settings.llama_base_url:
            raise ValueError("LLAMA_BASE_URL não configurada no .env.")
        self.base_url = settings.llama_base_url.rstrip("/")
        self.api_key = settings.llama_api_key or ""

    def generate(self, system: str, user_input: str, seed: Optional[int]) -> Tuple[str, dict]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_output_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        r = requests.post(url, headers=headers, json=payload, timeout=settings.timeout_s)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {}) or {}
        usage_dict = {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        return text, usage_dict


def build_provider() -> BaseProvider:
    """Instancia o provider definido no .env."""
    if settings.llm_provider == "openai":
        return OpenAIProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    if settings.llm_provider == "llama":
        return LlamaProvider()
    raise ValueError(f"LLM_PROVIDER inválido: {settings.llm_provider}")


class LLMOracle:
    """Oráculo semântico com cache, retries e self-consistency."""

    def __init__(self):
        root = Path(__file__).resolve().parent
        self.system_prompt = _load_text(root / "prompts" / "llm_system.txt")
        self.instructions = _load_text(root / "prompts" / "llm_instructions.txt")
        self.provider = build_provider()
        self.cache = SQLiteCache(settings.cache_path)
        self._last_error_short: str = ""  # para telemetria

    def _build_user_input(self, case: dict) -> str:
        """Monta o input do oráculo (instruções + payload JSON)."""
        payload = {
            "output_real": case.get("output_real", ""),
            "expected_claim": case.get("expected_claim", ""),
            "criteria": case.get("criteria", {}),
            "input_context": case.get("input_context", {}),
        }
        return self.instructions + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)

    def _call_provider(self, user_input: str, seed: Optional[int]) -> Tuple[str, dict, int, bool, int]:
        """
        Chama provider com cache e retries.
        Retorna: (text, usage, latency_ms, cache_hit, retries)
        """
        cache_payload = {
            "provider": self.provider.name,
            "model": settings.llm_model,
            "system": self.system_prompt,
            "user_input": user_input,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_output_tokens": settings.max_output_tokens,
            "seed": seed or 0,
        }
        key = SQLiteCache.make_key(cache_payload)

        cached = self.cache.get(key) if settings.enable_cache else None
        if cached:
            return (
                cached.get("text", ""),
                cached.get("usage", {}) or {},
                int(cached.get("latency_ms", 0) or 0),
                True,
                0,
            )

        self._last_error_short = ""
        last_err: Optional[Exception] = None
        retries = 0
        text = ""
        usage: dict = {}

        t0 = time.time()
        for attempt in range(settings.max_retries + 1):
            try:
                text, usage = self.provider.generate(self.system_prompt, user_input, seed)
                last_err = None
                break
            except Exception as e:
                # ✅ CORREÇÃO: mostrar o erro real no terminal (sem travar pipeline)
                last_err = e
                retries += 1

                err_short = f"{type(e).__name__}: {str(e)}"
                self._last_error_short = err_short[:300]

                print(
                    f"[LLM_CALL_FAILED] provider={self.provider.name} "
                    f"model={settings.llm_model} attempt={attempt+1}/{settings.max_retries+1} "
                    f"err={repr(e)}"
                )

                time.sleep(min(2**attempt, 8))

        latency_ms = int((time.time() - t0) * 1000)

        if last_err is not None:
            # mantém o pipeline andando (mas agora você viu o erro real no log)
            text = (
                '{"verdict":"INCONCLUSIVE","score":0.0,'
                '"reason":"llm_call_failed","evidence":[],"missing_requirements":[]}'
            )
            usage = {}

        if settings.enable_cache:
            self.cache.set(key, {"text": text, "usage": usage, "latency_ms": latency_ms})

        return text, usage, latency_ms, False, retries

    def evaluate(self, case: dict) -> OracleDecision:
        """
        Avalia um caso via LLM (semântica).
        Se self_consistency_n > 1, faz voto majoritário.
        """
        n = max(1, int(settings.self_consistency_n))
        user_input = self._build_user_input(case)

        verdict_counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
        scores: List[float] = []
        reasons: List[str] = []
        evidences: List[str] = []
        missing: List[str] = []

        total_in = 0
        total_out = 0
        retries_total = 0
        cache_hits = 0
        best_latency_ms: Optional[int] = None

        for _ in range(n):
            seed = random.randint(1, 10_000_000) if n > 1 else None
            text, usage, latency_ms, cache_hit, retries = self._call_provider(user_input, seed)

            cache_hits += int(cache_hit)
            retries_total += retries
            if best_latency_ms is None or latency_ms < best_latency_ms:
                best_latency_ms = latency_ms

            obj = _safe_parse_json(text) or {
                "verdict": "INCONCLUSIVE",
                "score": 0.0,
                "reason": "invalid_json",
                "evidence": [],
                "missing_requirements": [],
            }

            try:
                _validate_schema(obj)
            except Exception:
                obj = {
                    "verdict": "INCONCLUSIVE",
                    "score": 0.0,
                    "reason": "schema_invalid",
                    "evidence": [],
                    "missing_requirements": [],
                }

            v = str(obj.get("verdict", "INCONCLUSIVE")).upper()
            if v not in verdict_counts:
                v = "INCONCLUSIVE"
            verdict_counts[v] += 1

            s = float(obj.get("score", 0.0) or 0.0)
            s = max(0.0, min(1.0, s))
            scores.append(s)

            reasons.append(str(obj.get("reason", ""))[:250])

            for ev in (obj.get("evidence", []) or []):
                if isinstance(ev, str) and ev.strip():
                    evidences.append(ev.strip())

            for mr in (obj.get("missing_requirements", []) or []):
                if isinstance(mr, str) and mr.strip():
                    missing.append(mr.strip())

            total_in += int(usage.get("input_tokens", 0) or 0)
            total_out += int(usage.get("output_tokens", 0) or 0)

        majority_verdict = max(verdict_counts.items(), key=lambda kv: kv[1])[0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        est_cost = estimate_cost_usd(settings.llm_model, total_in, total_out)

        telemetry = {
            "oracle": "llm",
            "provider": self.provider.name,
            "model": settings.llm_model,
            "self_consistency_n": n,
            "latency_ms_best": int(best_latency_ms or 0),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "estimated_cost_usd": float(est_cost),
            "cache_hits": cache_hits,
            "retries": retries_total,
        }

        # ✅ ajuda a debugar em resultados_llm.jsonl sem olhar o terminal
        if self._last_error_short:
            telemetry["last_error"] = self._last_error_short

        return OracleDecision(
            verdict=majority_verdict,
            score=float(avg_score),
            reason="; ".join([r for r in reasons if r])[:800] or "no_reason",
            evidence=list(dict.fromkeys(evidences))[:6],
            missing_requirements=list(dict.fromkeys(missing))[:6],
            telemetry=telemetry,
        )
