from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 1) Carrega o .env ANTES de ler qualquer variável de ambiente
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Provider e modelo
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

    # Chaves
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    llama_base_url: str | None = os.getenv("LLAMA_BASE_URL")
    llama_api_key: str | None = os.getenv("LLAMA_API_KEY")

    # Runtime
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("LLM_TOP_P", "1.0"))
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500"))
    self_consistency_n: int = int(os.getenv("LLM_SELF_CONSISTENCY_N", "1"))

    # Cache
    enable_cache: bool = os.getenv("LLM_ENABLE_CACHE", "true").lower() == "true"
    cache_path: str = os.getenv("LLM_CACHE_PATH", ".llm_cache.sqlite")

    # Execução
    results_dir: str = os.getenv("RESULTS_DIR", "results")
    run_id: str = os.getenv("RUN_ID", "RUN-LOCAL-001")

    # Auto-geração
    auto_generate_tests: bool = os.getenv("AUTO_GENERATE_TESTS", "true").lower() == "true"
    auto_tests_n: int = int(os.getenv("AUTO_TESTS_N", "40"))

    # Sugestões de prompt
    prompt_improvements: bool = os.getenv("PROMPT_IMPROVEMENTS", "true").lower() == "true"
    prompt_improvements_max_cases: int = int(os.getenv("PROMPT_IMPROVEMENTS_MAX_CASES", "10"))


settings = Settings()
