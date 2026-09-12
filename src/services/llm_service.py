"""LLM wrapper for gpt-5.4 with reasoning_effort and usage tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class UsageStats:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.details.clear()

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0, note: str = "") -> None:
        self.llm_calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.details.append(
            {
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "note": note,
            }
        )


def _extract_usage(msg: Any) -> tuple[int, int]:
    """Best-effort token usage from LangChain AIMessage / dict."""
    if msg is None:
        return 0, 0
    meta = getattr(msg, "usage_metadata", None) or {}
    if isinstance(meta, dict) and meta:
        inn = meta.get("input_tokens") or meta.get("prompt_tokens") or 0
        out = meta.get("output_tokens") or meta.get("completion_tokens") or 0
        return int(inn or 0), int(out or 0)
    resp_meta = getattr(msg, "response_metadata", None) or {}
    if isinstance(resp_meta, dict):
        usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
        if isinstance(usage, dict):
            inn = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            return int(inn or 0), int(out or 0)
    if isinstance(msg, dict):
        um = msg.get("usage_metadata") or {}
        inn = um.get("input_tokens") or 0
        out = um.get("output_tokens") or 0
        return int(inn or 0), int(out or 0)
    return 0, 0


def uses_reasoning_api(model: str, *, use_reasoning: bool | None = None) -> bool:
    """Whether to use OpenAI Responses API + reasoning_effort.

    GPT-4.1 / GPT-4o-style chat models must NOT use reasoning_effort.
    GPT-5.x / o-series reasoning models do.
    """
    if use_reasoning is False:
        return False
    if use_reasoning is True:
        return True
    m = (model or "").lower().strip()
    if not m:
        return True
    if m.startswith("gpt-4.1") or m.startswith("gpt-4o") or m.startswith("gpt-4-"):
        return False
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    # Default: treat unknown OpenAI chat models as non-reasoning unless flagged
    return False


class LLMService:
    def __init__(
        self,
        model: str = "gpt-5.4",
        reasoning_effort: str = "low",
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        use_reasoning: bool | None = None,
        reasoning_via_chat_template: bool = False,
    ):
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.usage = UsageStats()
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if base_url:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        # Local OpenAI-compatible servers (vLLM) do not use Responses API / reasoning_effort
        use_local = bool(base_url)
        use_reason = (not use_local) and uses_reasoning_api(model, use_reasoning=use_reasoning)
        if use_local and disable_thinking:
            # Qwen3 chat template: turn off chain-of-thought / thinking blocks
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif use_local and reasoning_via_chat_template and reasoning_effort:
            # gpt-oss (harmony) chat template: keep reasoning on at the requested effort
            kwargs["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": reasoning_effort}}
        try:
            if use_local or not use_reason:
                self.llm = ChatOpenAI(**kwargs)
            else:
                self.llm = ChatOpenAI(
                    **kwargs,
                    reasoning_effort=reasoning_effort,
                    use_responses_api=True,
                )
        except TypeError:
            try:
                if use_local or not use_reason:
                    self.llm = ChatOpenAI(**kwargs)
                else:
                    self.llm = ChatOpenAI(**kwargs, reasoning_effort=reasoning_effort)
            except TypeError:
                self.llm = ChatOpenAI(**kwargs)

    def reset_usage(self) -> None:
        self.usage.reset()

    def _record_raw(self, raw: Any, note: str = "") -> None:
        inn, out = _extract_usage(raw)
        self.usage.add(input_tokens=inn, output_tokens=out, note=note)

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        repair: bool = True,
    ) -> T:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Prefer include_raw so we can capture token usage
        try:
            structured = self.llm.with_structured_output(schema, include_raw=True)
            use_raw = True
        except TypeError:
            structured = self.llm.with_structured_output(schema)
            use_raw = False

        def _parse(result: Any) -> T:
            if use_raw and isinstance(result, dict):
                raw = result.get("raw")
                parsed = result.get("parsed")
                self._record_raw(raw, note=schema.__name__)
                if isinstance(parsed, schema):
                    return parsed
                if parsed is not None:
                    return schema.model_validate(parsed)
                err = result.get("parsing_error") or "structured output parse failed"
                raise ValueError(err)
            if isinstance(result, schema):
                self.usage.add(note=schema.__name__)
                return result
            self.usage.add(note=schema.__name__)
            return schema.model_validate(result)

        try:
            result = structured.invoke(messages)
            return _parse(result)
        except Exception as e:
            if not repair:
                raise
            repair_prompt = (
                f"Your previous output was invalid for schema {schema.__name__}.\n"
                f"Error: {e}\n"
                f"Return a corrected JSON object matching the schema exactly.\n\n"
                f"Original task:\n{user}"
            )
            result = structured.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": repair_prompt},
                ]
            )
            return _parse(result)

    def raw_json(self, system: str, user: str) -> dict:
        resp = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user + "\n\nReturn JSON only."},
            ]
        )
        self._record_raw(resp, note="raw_json")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
