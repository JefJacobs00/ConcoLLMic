"""
For models other than those from OpenAI, use LiteLLM if possible.
"""

import os
import sys
import time
from typing import Literal

import litellm
from litellm import completion_cost
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from litellm.utils import Choices, Message, ModelResponse
from loguru import logger

from app.log import log_and_print, print_usage_compact
from app.model import common
from app.model.common import Model, ModelNoResponseError, Usage


class GeminiModel(Model):
    """
    Base class for creating Singleton instances of Gemini models.
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
        max_output_token: int = 4096,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self.max_output_token = max_output_token
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        key_name = "GEMINI_API_KEY"
        key = os.getenv(key_name)
        if not key:
            print(f"Please set the {key_name} env var")
            sys.exit(1)
        return key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
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
            if response_format == "json_object":
                last_content = messages[-1]["content"]
                last_content += "\nYour response should start with { and end with }. DO NOT write anything else other than the json."
                messages[-1]["content"] = last_content

            start_time = time.time()
            response = litellm.completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_output_token,
                top_p=top_p,
                stream=False,
                tools=tools,
                **kwargs,
            )

            latency = time.time() - start_time
            cost = completion_cost(model=self.name, completion_response=response)

            assert isinstance(response, ModelResponse)

            # Check if the response has valid choices before proceeding
            if not response.choices or len(response.choices) == 0:
                raise ModelNoResponseError(
                    f"Model {self.name} returned a response with no choices. Response: {response}"
                )

            resp_usage = response.usage
            assert resp_usage is not None

            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)

            cache_creation_tokens = int(
                resp_usage.get("cache_creation_input_tokens", 0)
            )
            cache_read_tokens = int(resp_usage.get("cache_read_input_tokens", 0))

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message

            # Extract content from the message
            content = self.extract_resp_content(resp_msg)

            # Extract tool calls from the message
            tool_calls = (
                resp_msg.tool_calls if hasattr(resp_msg, "tool_calls") else None
            )

            logger.info(
                f"Model ({self.name}) API request usage info: "
                f"{{input_tokens={input_tokens}, output_tokens={output_tokens}, cache_read_tokens={cache_read_tokens}, cache_write_tokens={cache_creation_tokens}}}, cost={cost:.6f} USD, latency={latency:.6f} seconds",
            )
            print_usage_compact(self.name, cost, latency)

            return (
                content,
                tool_calls,
                Usage(
                    model=self.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_creation_tokens,
                    cost=cost,
                    latency=latency,
                    call_cnt=1,
                ),
            )

        except LiteLLMBadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


# We store the Standard tier in `cost_per_input` / `cost_per_output`.
# See https://ai.google.dev/gemini-api/docs/models for details.
class Gemini2_0Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-2.0-flash",
            0.0000001,
            0.0000004,
            parallel_tool_call=True,
            max_output_token=8192,
        )
        self.note = "Deprecated Gemini 2.0 Flash model via Google AI Studio; scheduled shutdown June 1, 2026"


class Gemini2_5Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-2.5-flash",
            0.0000003,
            0.0000025,
            parallel_tool_call=True,
            max_output_token=65536,
        )
        self.note = "Balanced Gemini 2.5 Flash model via Google AI Studio"


class Gemini2_5Pro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-2.5-pro",
            0.00000125,
            0.00001,
            parallel_tool_call=True,
            max_output_token=65536,
        )
        self.note = "Most capable Gemini 2.5 model via Google AI Studio"


class Gemini3Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-3-flash-preview",
            0.0000005,
            0.000003,
            parallel_tool_call=True,
            max_output_token=65536,
        )
        self.note = "Preview Gemini 3 Flash model via Google AI Studio"


class Gemini3_1Pro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-3.1-pro-preview",
            0.000002,
            0.000012,
            parallel_tool_call=True,
            max_output_token=65536,
        )
        self.note = "Preview Gemini 3.1 Pro model via Google AI Studio"


class Gemini3_5Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-3.5-flash",
            0.0000015,
            0.000009,
            parallel_tool_call=True,
            max_output_token=65536,
        )
        self.note = "Stable Gemini 3.5 Flash model via Google AI Studio"
