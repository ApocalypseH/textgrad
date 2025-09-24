try:
    from openai import AzureOpenAI, OpenAI, NotGiven, NOT_GIVEN, DEFAULT_MAX_RETRIES
except ImportError:
    raise ImportError(
        "If you'd like to use customized API models, please install the openai package by running `pip install openai`."
    )

import base64
import json
import os
import re
from typing import List, Union, Optional

import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import CachedEngine, EngineLM
from .engine_utils import get_image_type_from_bytes

# Default base URL for OLLAMA
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


class BaseOpenAIEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        cache_path: str,
        system_prompt: str,
        model_string: str,
        is_multimodal: bool = False,
        reasoning_effort: Union[str, NotGiven] = NOT_GIVEN,
        stream: bool = False,
    ):
        super().__init__(cache_path=cache_path)
        self.system_prompt = system_prompt
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.reasoning_effort = reasoning_effort
        self.stream = stream

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str = None,
        **kwargs,
    ):
        if isinstance(content, str):
            return self._generate_from_single_prompt(
                content, system_prompt=system_prompt, **kwargs
            )

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError(
                    "Multimodal generation is only supported for Claude-3 and beyond."
                )

            return self._generate_from_multiple_input(
                content, system_prompt=system_prompt, **kwargs
            )

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=10000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        if self.stream:
            for attempt in range(DEFAULT_MAX_RETRIES + 1):
                response_text = ""
                response = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=[
                        {"role": "system", "content": sys_prompt_arg},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning_effort=self.reasoning_effort,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                )

                try:
                    for chunk in response:
                        response_text += chunk.choices[0].delta.content
                        if chunk.choices[0].finish_reason is not None:
                            break
                    break  # If we reach here, the streaming was successful, so we break out of the retry loop
                except Exception as e:
                    if attempt == DEFAULT_MAX_RETRIES:
                        raise e
        else:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort=self.reasoning_effort,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_text = response.choices[0].message.content

        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if len(response_text) > 4000:
            response_text = response_text[:2000] + "..." + response_text[-2000:]
        self._save_cache(sys_prompt_arg + prompt, response_text)
        return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API."""
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode("utf-8")
                formatted_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}"
                        },
                    }
                )
            elif isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=10000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none
        
        if self.stream:
            for attempt in range(DEFAULT_MAX_RETRIES + 1):
                response_text = ""
                response = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=[
                        {"role": "system", "content": sys_prompt_arg},
                        {"role": "user", "content": formatted_content},
                    ],
                    reasoning_effort=self.reasoning_effort,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                )

                try:
                    for chunk in response:
                        response_text += chunk.choices[0].delta.content
                        if chunk.choices[0].finish_reason is not None:
                            break
                    break  # If we reach here, the streaming was successful, so we break out of the retry loop
                except Exception as e:
                    if attempt == DEFAULT_MAX_RETRIES:
                        raise e
        else:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
                reasoning_effort=self.reasoning_effort,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_text = response.choices[0].message.content

        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if len(response_text) > 4000:
            response_text = response_text[:2000] + "..." + response_text[-2000:]
        self._save_cache(cache_key, response_text)
        return response_text


class ChatOpenAI(BaseOpenAIEngine):
    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo-0613",
        system_prompt: str = BaseOpenAIEngine.DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Union[str, NotGiven] = NOT_GIVEN,
        cache_root: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support customized API service, if not provided, it will look for OPENAI_BASE_URL in environment variables
        :param api_key: API key for authentication, if not provided, it will look for OPENAI_API_KEY in environment variables
        """
        if cache_root is None:
            root = platformdirs.user_cache_dir("textgrad")
        else:
            root = cache_root
            os.mkdir(root) if not os.path.exists(root) else None
        cache_path = os.path.join(root, f"cache_my_{model_string}.db")

        super().__init__(cache_path, system_prompt, model_string, is_multimodal, reasoning_effort, stream)

        if not base_url and os.getenv("OPENAI_BASE_URL") is None:
            raise ValueError(
                "Please set the OPENAI_BASE_URL environment variable or pass `base_url` if you'd like to use customized API service."
            )
        base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")

        self.base_url = base_url

        if not api_key and os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable or pass `api_key` if you'd like to use customized API models."
            )
        api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
