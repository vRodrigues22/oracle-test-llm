from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from config import settings
from llm_oracle import LLMOracle
from traditional_oracle import evaluate as trad_eval


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def smoke_test(llm: LLMOracle) -> None:
    """Testa se o provider atual está respondendo."""
    case = {
        "id": "SMOKE-001",
        "output_real": "Não foi possível entrar. Usuário ou senha incorretos.",
        "expected_claim": "A mensagem indica credenciais inválidas e é apropriada para o usuário final.",
        "criteria": {"semantic_equivalence": True, "appropriateness": True, "must_not_include": ["stack trace"]},
        "input_context": {"language": "pt-BR", "user_type": "end_user", "severity": "low"},
    }
    d = llm.evaluate(case)
    print("[SMOKE] provider =", d.telemetry.get("provider"), "model =", d.telemetry.get("model"))
    print("[SMOKE] verdict =", d.verdict, "score =", d.score)
    print("[SMOKE] reason =", d.reason)


def generate_synthetic_tests(n: int) -> Tuple[List[dict], Dict[str, str]]:
    """
    Sistema auto-imbutido (versão determinística):
    - cria casos PASS/FAIL por construção, para ter ground truth confiável
    - você pode expandir categorias facilmente
    """
    templates_pass = [
        ("Não foi possível entrar. Usuário ou senha incorretos.", "Credenciais inválidas."),
        ("Acesso negado: credenciais inválidas.", "Credenciais inválidas."),
        ("Falha no login: usuário/senha incorretos.", "Credenciais inválidas."),
    ]
    templates_fail = [
        ("Login realizado com sucesso.", "Credenciais inválidas."),
        ("Sistema indisponível no momento. Tente mais tarde.", "Credenciais inválidas."),
        ("Sua senha foi alterada com sucesso.", "Credenciais inválidas."),
    ]

    cases: List[dict] = []
    gt: Dict[str, str] = {}

    for i in range(n):
        is_pass = (i % 2 == 0)
        if is_pass:
            out, _ = templates_pass[i % len(templates_pass)]
            cid = f"AUTO-LOGIN-PASS-{i:03d}"
            gt[cid] = "PASS"
        else:
            out, _ = templates_fail[i % len(templates_fail)]
            cid = f"AUTO-LOGIN-FAIL-{i:03d}"
            gt[cid] = "FAIL"

        case = {
            "id": cid,
            "category": "auth.login",
            "input_context": {"language": "pt-BR", "user_type": "end_user", "severity": "medium"},
            "output_real": out,
            "expected_claim": "A operação falhou devido a credenciais inválidas e a mensagem é apropriada para o usuário final.",
            "criteria": {
                "semantic_equivalence": True,
                "appropriateness": True,
                "must_not_include": ["stack trace", "Exception:", "token", "senha:"],
            },
        }
        cases.append(case)

    return cases, gt


def build_report(
    run_id: str,
    provider: str,
    model: str,
    metrics_trad: dict,
    metrics_llm: dict,
    llm_rows: List[dict],
    gt: Dict[str, str],
    prompt_suggestions: str | None,
) -> str:
    """Gera relatório em Markdown."""
    # Diagnóstico de erros (FP/FN) para análise
    failures: List[dict] = []
    for row in llm_rows:
        cid = row["case_id"]
        pred = row["decision"]["verdict"].upper()
        true = gt.get(cid, "UNKNOWN")
        if true in ("PASS", "FAIL") and pred in ("PASS", "FAIL") and pred != true:
            failures.append({
                "case_id": cid,
                "true": true,
                "pred": pred,
                "reason": row["decision"].get("reason", ""),
                "output_real": row.get("case", {}).get("output_real", ""),
                "expected_claim": row.get("case", {}).get("expected_claim", ""),
                "criteria": row.get("case", {}).get("criteria", {}),
            })

    lines = []
    lines.append(f"# Relatório — {run_id}")
    lines.append("")
    lines.append(f"- Provider: **{provider}**")
    lines.append(f"- Model: **{model}**")
    lines.append(f"- Timestamp (UTC): {now_utc_iso()}")
    lines.append("")

    lines.append("## Métricas — Oráculo Tradicional")
    lines.append("```json")
    lines.append(json.dumps(metrics_trad, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Métricas — Oráculo LLM (Semântico)")
    lines.append("```json")
    lines.append(json.dumps(metrics_llm, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Principais Erros do LLM (amostra)")
    if failures:
        for f in failures[:10]:
            lines.append(f"### {f['case_id']} (true={f['true']} pred={f['pred']})")
            lines.append(f"- reason: {f['reason']}")
            lines.append(f"- output_real: {f['output_real']}")
            lines.append(f"- expected_claim: {f['expected_claim']}")
            lines.append(f"- criteria: {json.dumps(f['criteria'], ensure_ascii=False)}")
            lines.append("")
    else:
        lines.append("Nenhum erro PASS/FAIL detectado (ou apenas INCONCLUSIVE).")
        lines.append("")

    if prompt_suggestions:
        lines.append("## Sugestões de Melhoria de Prompt")
        lines.append(prompt_suggestions)
        lines.append("")

    lines.append("## Observações")
    lines.append("- INCONCLUSIVE não entra na acurácia (por design).")
    lines.append("- Para reduzir FP: reforçar must_not_include e exigir evidências explícitas.")
    lines.append("- Para reduzir FN: permitir variações linguísticas e sinônimos, mantendo regras determinísticas.")
    lines.append("")
    return "\n".join(lines)


def compute_metrics_from_results(gt: Dict[str, str], rows: List[dict]) -> dict:
    """Métricas internas (mesma lógica do metrics.py), para compor relatório automaticamente."""
    tp = tn = fp = fn = inconc = total = 0

    for r in rows:
        cid = r["case_id"]
        pred = r["decision"]["verdict"].upper()
        true = gt.get(cid)
        if true is None:
            continue
        total += 1
        if pred == "INCONCLUSIVE":
            inconc += 1
            continue
        if true == "PASS" and pred == "PASS":
            tp += 1
        elif true == "FAIL" and pred == "FAIL":
            tn += 1
        elif true == "FAIL" and pred == "PASS":
            fp += 1
        elif true == "PASS" and pred == "FAIL":
            fn += 1

    denom = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / denom
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    inconc_rate = inconc / total if total else 0.0

    return {
        "total_evaluated": total,
        "inconclusive_rate": inconc_rate,
        "accuracy_excl_inconclusive": accuracy,
        "precision_PASS": precision,
        "recall_PASS": recall,
        "f1_PASS": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "inconclusive": inconc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Executa apenas teste de integração do provider atual.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    ensure_dir(data_dir)

    results_dir = root / settings.results_dir
    ensure_dir(results_dir)

    llm = LLMOracle()

    if args.smoke:
        smoke_test(llm)
        return

    # 1) Auto-gerar dataset (opcional) — sistema auto-imbutido
    if settings.auto_generate_tests:
        cases, gt = generate_synthetic_tests(settings.auto_tests_n)
        save_json(data_dir / "test_cases.json", cases)
        save_json(data_dir / "ground_truth.json", gt)
    else:
        cases = load_json(data_dir / "test_cases.json")
        gt = load_json(data_dir / "ground_truth.json")
        gt = {k: str(v).upper() for k, v in gt.items()}

    # 2) Executar oráculos
    llm_rows: List[Dict[str, Any]] = []
    trad_rows: List[Dict[str, Any]] = []

    for case in tqdm(cases, desc="Running cases"):
        cid = case.get("id", "UNKNOWN")

        # baseline tradicional
        t = trad_eval(case, mode="regex")
        trad_rows.append({
            "run_id": settings.run_id,
            "oracle": "traditional",
            "case_id": cid,
            "case": case,
            "decision": {"verdict": t.verdict, "score": t.score, "reason": t.reason},
            "telemetry": t.telemetry,
            "timestamp_utc": now_utc_iso(),
        })

        # oráculo LLM
        d = llm.evaluate(case)
        llm_rows.append({
            "run_id": settings.run_id,
            "oracle": "llm",
            "case_id": cid,
            "case": case,
            "decision": {
                "verdict": d.verdict,
                "score": d.score,
                "reason": d.reason,
                "evidence": d.evidence,
                "missing_requirements": d.missing_requirements,
            },
            "telemetry": d.telemetry,
            "timestamp_utc": now_utc_iso(),
        })

    # 3) Salvar resultados
    write_jsonl(results_dir / "results_traditional.jsonl", trad_rows)
    write_jsonl(results_dir / "results_llm.jsonl", llm_rows)

    # 4) Métricas internas (para relatório auto)
    m_trad = compute_metrics_from_results(gt, trad_rows)
    m_llm = compute_metrics_from_results(gt, llm_rows)

    # 5) Sugestões de melhoria de prompt (opcional)
    prompt_suggestions = None
    if settings.prompt_improvements:
        # Coleta erros do LLM (FP/FN) como base para melhorias
        failures = []
        for row in llm_rows:
            cid = row["case_id"]
            pred = row["decision"]["verdict"].upper()
            true = gt.get(cid, "UNKNOWN")
            if true in ("PASS", "FAIL") and pred in ("PASS", "FAIL") and pred != true:
                failures.append({
                    "case_id": cid,
                    "true": true,
                    "pred": pred,
                    "llm_reason": row["decision"].get("reason", ""),
                    "output_real": row["case"].get("output_real", ""),
                    "expected_claim": row["case"].get("expected_claim", ""),
                    "criteria": row["case"].get("criteria", {}),
                })

        sys_prompt = (root / "prompts" / "llm_system.txt").read_text(encoding="utf-8")
        instr_prompt = (root / "prompts" / "llm_instructions.txt").read_text(encoding="utf-8")

        if failures:
            prompt_suggestions = llm.suggest_prompt_improvements(
                failures=failures,
                current_system=sys_prompt,
                current_instructions=instr_prompt,
                max_cases=settings.prompt_improvements_max_cases,
            )
        else:
            prompt_suggestions = "Não houve FP/FN suficientes para sugerir alterações (ou apenas INCONCLUSIVE)."

    # 6) Relatório
    provider = llm.provider.name
    model = settings.llm_model
    report = build_report(
        run_id=settings.run_id,
        provider=provider,
        model=model,
        metrics_trad=m_trad,
        metrics_llm=m_llm,
        llm_rows=llm_rows,
        gt=gt,
        prompt_suggestions=prompt_suggestions,
    )
    report_path = results_dir / f"report_{settings.run_id}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[OK] wrote {len(trad_rows)} -> {results_dir / 'results_traditional.jsonl'}")
    print(f"[OK] wrote {len(llm_rows)} -> {results_dir / 'results_llm.jsonl'}")
    print(f"[OK] wrote report -> {report_path}")


if __name__ == "__main__":
    main()
