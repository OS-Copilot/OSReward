# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        return output.text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']

    if 'OPENAI_BASE_URL' in os.environ:
      self.base_url = os.environ['OPENAI_BASE_URL']
    else:
      self.base_url = 'https://api.openai.com/'
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        # response = requests.post(
        #     'https://api.openai.com/v1/chat/completions',
        #     headers=headers,
        #     json=payload,
        # )
        response = requests.post(
            f'{self.base_url}v1/chat/completions',
            headers=headers,
            json=payload,
        )
        if response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None


import base64
from typing import Any, Optional
import numpy as np
class Claude4WrapperV1(LlmWrapper, MultimodalLlmWrapper):
  """Anthropic Claude wrapper using requests (REST API)."""

  RETRY_WAITING_SECONDS = 10

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    
    if 'ANTHROPIC_API_KEY' not in os.environ:
      raise RuntimeError('Anthropic API key not set.')
    self.api_key = os.environ['ANTHROPIC_API_KEY']

    if 'ANTHROPIC_BASE_URL' in os.environ:
      self.base_url = os.environ['ANTHROPIC_BASE_URL']
    else:
      self.base_url = 'https://api.anthropic.com/'
    
    # Ensure base_url ends with slash for cleaner appending
    if not self.base_url.endswith('/'):
        self.base_url += '/'

    self.model_name = model_name
    self.temperature = temperature
    
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    """Encodes a numpy image to a base64 string."""
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    
    headers = {
        'x-api-key': self.api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    # Construct the content blocks
    content_blocks = []
    
    # Add images first
    for image in images:
      content_blocks.append({
          "type": "image",
          "source": {
              "type": "base64",
              "media_type": "image/jpeg",
              "data": self.encode_image(image),
          }
      })
    
    # Add text prompt
    content_blocks.append({
        "type": "text",
        "text": text_prompt
    })

    payload = {
        "model": self.model_name,
        "max_tokens": 1024,
        "temperature": self.temperature,
        "messages": [
            {
                "role": "user",
                "content": content_blocks
            }
        ]
    }

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS

    while counter > 0:
      try:
        # Using v1/messages as per standard Anthropic API
        response = requests.post(
            f'{self.base_url}v1/messages',
            headers=headers,
            json=payload,
        )

        if response.ok:
          response_json = response.json()
          # Parse Anthropic response format
          # { "content": [ { "text": "..." } ] }
          if 'content' in response_json and len(response_json['content']) > 0:
            text_output = response_json['content'][0].get('text', '')
            return text_output, True, response_json
        
        # Handle errors
        print(
            f'Error calling Claude API: {response.status_code} - '
            + response.text
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1

      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print(f'Error calling Claude LLM, will retry in {wait_seconds} seconds')
        print(e)
        if counter > 0:
          time.sleep(wait_seconds)
          wait_seconds *= 2

    return ERROR_CALLING_LLM, None, None


from android_world.api.llm_api_utils import get_model_response
class Claude4WrapperV2(LlmWrapper, MultimodalLlmWrapper):
  """
  Claude wrapper speaking the OpenAI chat-completion protocol.

  Adapts the OpenAI-style response back into the raw Anthropic JSON format so
  the raw_response field stored in trajectories keeps a consistent shape.
  """

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
      timeout: int = 60,
  ):
    if 'ANTHROPIC_API_KEY' not in os.environ:
      raise RuntimeError('Anthropic API key not set.')
    self.api_key = os.environ['ANTHROPIC_API_KEY']

    if 'ANTHROPIC_BASE_URL' in os.environ:
      self.base_url = os.environ['ANTHROPIC_BASE_URL']
    else:
      self.base_url = 'https://api.anthropic.com/'
    
    # Ensure base_url ends with slash
    if not self.base_url.endswith('/'):
        self.base_url += '/'

    self.model_name = model_name
    self.temperature = temperature
    self.timeout = timeout
    
    if max_retry <= 0:
      max_retry = 3
    self.max_retry = min(max_retry, 5)

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    """Encodes a numpy image to a base64 Data URI string."""
    image_bytes = array_to_jpeg_bytes(image) 
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    
    # 1. Construct Content Payload (User Message)
    content_payload = []

    # Add text
    content_payload.append({
        "type": "text",
        "text": text_prompt
    })
    
    # Add images
    for image in images:
      base64_image_url = self.encode_image(image)
      content_payload.append({
          "type": "image_url",
          "image_url": {
              "url": base64_image_url
          }
      })

    # 2. Construct Messages List
    messages = [
        {"role": "system", "content": "You are a helpful assistant capable of analyzing images."},
        {"role": "user", "content": content_payload}
    ]

    try:
        # 3. Call the model
        completion = get_model_response(
            model_url=f"{self.base_url}v1",
            model_name=self.model_name,
            model_token=self.api_key,
            messages=messages,
            tool_schemas=None,
            agent_logger=None,
            max_retry_num=self.max_retry,
            temperature=self.temperature,
            max_tokens=1024,
            timeout=self.timeout,
        )
        text_output = completion.choices[0].message.content

        # 4. Reconstruct an Anthropic-shaped response envelope.
        response_json = {
            "id": completion.id,
            "type": "message",
            "role": "assistant",
            "model": self.model_name,
            "content": [
                {
                    "type": "text",
                    "text": text_output
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
            }
        }

        return text_output, True, response_json

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "ERROR_CALLING_LLM", None, None