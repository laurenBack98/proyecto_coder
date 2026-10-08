import inspect
from typing import AsyncGenerator, List

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APIError as AnthropicAPIError,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.resources.messages import AsyncMessages

from llm_client.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelResponse, Provider


# Las versiones recientes del SDK de Anthropic ya no exponen `temperature` en messages.create.
# Lo enviamos solo si el SDK instalado lo acepta; si no, se omite para no romper la llamada.
_SDK_ACEPTA_TEMPERATURE = "temperature" in inspect.signature(AsyncMessages.create).parameters


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def _build_sdk_client(self, api_key: str) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=api_key)

    def _kwargs(self, messages: List[ChatMessage]) -> dict:
        """Anthropic recibe el system como parámetro aparte, no dentro de messages."""
        system, resto = self._separar_system(messages)
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
            messages=[m.model_dump() for m in resto],
        )
        if _SDK_ACEPTA_TEMPERATURE:
            kwargs["temperature"] = self.temperature
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._kwargs(messages))
            return ModelResponse(
                provider=self.provider,
                model=self.model,
                content=response.content[0].text,
            )
        except AnthropicRateLimitError as e:
            return self._error_response(f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return self._error_response(f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return self._error_response(f"Error de la API de Anthropic: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._kwargs(messages)) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
