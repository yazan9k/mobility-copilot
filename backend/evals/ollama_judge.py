"""A DeepEval judge backed by a local Ollama model.

DeepEval defaults to OpenAI. This class replaces that, which is mandatory for
a zero-cost setup, not an optimisation.

The important detail is `generate_with_schema`. DeepEval hands custom models a
Pydantic schema and expects either an instance of it back or JSON it can parse.
Left to free-form generation, a 14B model produces unparseable or
wrongly-shaped output often enough to make LLM-scored metrics unusable — the
scores that survive are biased toward the cases where the model happened to
comply. Passing the schema to Ollama's `format` parameter constrains decoding
at the token level, so the output is structurally valid by construction.

That is what makes local judging viable here. It does not make the judge
*accurate* — accuracy is measured separately by evals/calibration.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Type, TypeVar

import ollama
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from config import JUDGE_MODEL, OLLAMA_HOST

T = TypeVar("T", bound=BaseModel)


class OllamaJudge(DeepEvalBaseLLM):
    """Local judge model for G-Eval and faithfulness scoring."""

    def __init__(self, model_name: str | None = None, temperature: float = 0.0):
        self.model_name = model_name or JUDGE_MODEL
        self.temperature = temperature
        self._client = ollama.Client(host=OLLAMA_HOST)
        super().__init__(self.model_name)

    # --- DeepEvalBaseLLM contract -----------------------------------------

    def load_model(self, *args: Any, **kwargs: Any) -> "OllamaJudge":
        return self

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"ollama/{self.model_name}"

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_temperature(self) -> bool:
        return True

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        schema = kwargs.get("schema")
        if schema is not None:
            return self.generate_with_schema(prompt, schema=schema)
        return self._raw(prompt, fmt=None)

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, prompt, *args, **kwargs)

    def generate_with_schema(
        self, prompt: str, schema: Type[T] | None = None, *args: Any, **kwargs: Any
    ) -> T | str:
        if schema is None:
            return self._raw(prompt, fmt=None)

        text = self._raw(prompt, fmt=schema.model_json_schema())
        try:
            return schema.model_validate_json(text)
        except Exception:
            # Constrained decoding makes this rare, but a malformed response
            # should surface as a scoring failure rather than a crash mid-run.
            # Returning the raw string lets DeepEval attempt its own parse.
            return text

    async def a_generate_with_schema(
        self, prompt: str, schema: Type[T] | None = None, *args: Any, **kwargs: Any
    ) -> T | str:
        return await asyncio.to_thread(
            self.generate_with_schema, prompt, schema, *args, **kwargs
        )

    # --- internals ---------------------------------------------------------

    def _raw(self, prompt: str, fmt: dict | None) -> str:
        response = self._client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=fmt,
            options={"temperature": self.temperature},
        )
        return (response.get("message", {}) or {}).get("content", "") or ""


def health_check() -> tuple[bool, str]:
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        available = {m.get("model", "") for m in client.list().get("models", [])}
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"

    if JUDGE_MODEL not in available:
        return False, (
            f"Judge model {JUDGE_MODEL!r} not found. Run: ollama pull {JUDGE_MODEL}"
        )
    return True, f"Judge OK · {JUDGE_MODEL}"


if __name__ == "__main__":
    ok, msg = health_check()
    print(msg)
    if ok:

        class _Verdict(BaseModel):
            score: float
            reason: str

        judge = OllamaJudge()
        out = judge.generate_with_schema(
            "Rate how well this answers 'what is 2+2?': 'The answer is 4.' "
            "Return a score between 0 and 1 and a one-sentence reason.",
            schema=_Verdict,
        )
        print("schema-constrained output:", out)
        print("type:", type(out).__name__)
