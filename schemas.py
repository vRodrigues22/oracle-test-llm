# Schema JSON para validar a resposta do oráculo LLM.

ORACLE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "score", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "INCONCLUSIVE"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "missing_requirements": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}
