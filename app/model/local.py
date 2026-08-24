"""
Interfacing with locally-served, OpenAI-compatible models (vLLM, SGLang, llama.cpp, ...).

We go through LiteLLM (like `gemini.py`) rather than the native OpenAI client, because
the agents consume tool calls as dicts (`tool_call.get("function")`) while also handing
them to `MessageThread.add_model`, which uses attribute access. LiteLLM's message objects
support both.
"""

import json
import os
import time
import urllib.request
from typing import Literal

import litellm
from litellm import completion_cost
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from litellm.utils import Choices, Message, ModelResponse
from loguru import logger

from app.log import log_and_print, print_usage_compact
from app.model import common
from app.model.common import Model, ModelNoResponseError, Usage

DEFAULT_API_BASE = "http://localhost:8000/v1"
# Reasoning models spend part of this budget thinking before the tool call lands,
# so it needs headroom above what the visible answer costs.
DEFAULT_MAX_OUTPUT_TOKENS = 16384


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """
    Make the message thread palatable to a plain OpenAI-compatible server.

    Two transformations, neither of which changes semantics:
      - drop Anthropic-style `cache_control` markers, which local servers reject as
        unknown fields;
      - flatten single-part text content to a bare string, which every chat template
        handles, whereas content-part lists depend on the server's parser.
    """
    cleaned = []
    for msg in messages:
        new_msg = {k: v for k, v in msg.items() if k != "cache_control"}
        content = new_msg.get("content")

        if isinstance(content, list):
            parts = [
                ({k: v for k, v in part.items() if k != "cache_control"})
                if isinstance(part, dict)
                else part
                for part in content
            ]
            if (
                len(parts) == 1
                and isinstance(parts[0], dict)
                and parts[0].get("type") == "text"
            ):
                new_msg["content"] = parts[0].get("text")
            else:
                new_msg["content"] = parts

        cleaned.append(new_msg)
    return cleaned


class LocalOpenAICompatModel(Model):
    """
    A model served locally behind an OpenAI-compatible /v1/chat/completions endpoint.

    `served_name` must match what the server advertises (vLLM's `--served-model-name`,
    which defaults to the HuggingFace repo path).
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        served_name: str,
        api_base: str | None = None,
        max_output_length: int = DEFAULT_MAX_OUTPUT_TOKENS,
        context_length: int = 200_000,
        parallel_tool_call: bool = True,
    ):
        if self._initialized:
            return
        # Local serving is free; costs stay at zero so `run_data` reports 0 USD.
        super().__init__(f"openai/{served_name}", 0.0, 0.0, parallel_tool_call, max_output_length=max_output_length, context_length=context_length)
        self.served_name = served_name
        # Where the server lives is a deployment detail, not a model property.
        self.api_base = api_base or os.getenv("LOCAL_MODEL_API_BASE", DEFAULT_API_BASE)
        self.max_output_token = max_output_length
        self._initialized = True

    def setup(self) -> None:
        """
        Teach LiteLLM about this model so its cost lookups resolve to zero instead of
        raising. `common.get_usage_input_part` calls `cost_per_token` on every agent
        turn, so an unmapped name would abort the run.
        """
        zero_cost = {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "max_tokens": self.max_output_token,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        litellm.register_model({self.name: zero_cost, self.served_name: zero_cost})

        self.ctx_len = self._discover_ctx_len()

        logger.info(
            "Local model {} registered, serving from {} (context length: {})",
            self.name,
            self.api_base,
            self.ctx_len if self.ctx_len else "unknown",
        )

    def _discover_ctx_len(self) -> int | None:
        """
        Ask the server how long a sequence it will accept. Agents size their own
        history against this, so guessing wrong means a non-retryable 400 mid-run.
        """
        try:
            with urllib.request.urlopen(
                f"{self.api_base.rstrip('/')}/models", timeout=10
            ) as resp:
                models = json.load(resp).get("data", [])
            for entry in models:
                if entry.get("id") == self.served_name and entry.get("max_model_len"):
                    return int(entry["max_model_len"])
            # Single-model servers may advertise a different id than we request.
            if len(models) == 1 and models[0].get("max_model_len"):
                return int(models[0]["max_model_len"])
        except Exception as e:
            logger.warning("Could not determine context length from server: {}", e)
        return None

    def check_api_key(self) -> str:
        # Local servers are usually open; vLLM's --api-key is honoured if set.
        return os.getenv("LOCAL_MODEL_API_KEY", "EMPTY")

    def extract_resp_content(self, chat_message: Message) -> str:
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    def _perform_call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ):
        if temperature is None:
            temperature = common.MODEL_TEMP

        try:
            request_kwargs = dict(kwargs)

            # `any` is Anthropic's spelling; the OpenAI schema calls it `required`.
            if request_kwargs.get("tool_choice") == "any":
                request_kwargs["tool_choice"] = "required"

            if response_format == "json_object":
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    last_content[-1]["text"] += (
                        "\nYour response should start with { and end with }. "
                        "DO NOT write anything else other than the json."
                    )
                else:
                    messages[-1]["content"] = last_content + (
                        "\nYour response should start with { and end with }. "
                        "DO NOT write anything else other than the json."
                    )

            start_time = time.time()
            response = litellm.completion(
                model=self.served_name,
                custom_llm_provider="openai",
                api_base=self.api_base,
                api_key=self.check_api_key(),
                messages=_sanitize_messages(messages),
                temperature=temperature,
                max_tokens=self.max_output_token,
                top_p=top_p,
                stream=False,
                tools=tools,
                drop_params=True,  # tolerate servers lacking e.g. parallel_tool_calls
                **request_kwargs,
            )

            latency = time.time() - start_time

            assert isinstance(response, ModelResponse)

            if not response.choices or len(response.choices) == 0:
                raise ModelNoResponseError(
                    f"Model {self.name} returned a response with no choices. Response: {response}"
                )

            resp_usage = response.usage
            assert resp_usage is not None

            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)

            try:
                cost = completion_cost(model=self.name, completion_response=response)
            except Exception:  # unmapped local model: free by definition
                cost = 0.0

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message

            content = self.extract_resp_content(resp_msg)
            tool_calls = (
                resp_msg.tool_calls if hasattr(resp_msg, "tool_calls") else None
            )

            logger.info(
                f"Model ({self.name}) API request usage info: "
                f"{{input_tokens={input_tokens}, output_tokens={output_tokens}}}, "
                f"cost={cost:.6f} USD, latency={latency:.6f} seconds",
            )
            print_usage_compact(self.name, cost, latency)

            return (
                content,
                tool_calls,
                Usage(
                    model=self.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    cost=cost,
                    latency=latency,
                    call_cnt=1,
                ),
            )

        except LiteLLMBadRequestError as e:
            # Not retried by tenacity: usually the prompt outgrew --max-model-len.
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


class Qwen3_5_9B(LocalOpenAICompatModel):
    def __init__(self):
        super().__init__("Qwen/Qwen3.5-9B", max_output_length=32_768)
        self.note = "Qwen3.5 9B served locally. Hybrid attention, reasoning model."


class Qwen3_8_27B_NVFP4(LocalOpenAICompatModel):
    def __init__(self):
        super().__init__("gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090", max_output_length=32_768 , context_length=262144)
        self.note = "Qwen3.8 27B quantized to NVFP4, served locally."

class Qwen3_8_27B(LocalOpenAICompatModel):
    def __init__(self):
        super().__init__("Qwen/Qwen3.8-27B", max_output_length=32_768, context_length=262144)
        self.note = "Qwen3.8 27B at bf16/fp16 served locally. ~54 GB of weights."