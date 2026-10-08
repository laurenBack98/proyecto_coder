from typing import AsyncGenerator, List

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from llm_client.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def _build_sdk_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],  # OpenAI acepta role="system" tal cual
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return ModelResponse(
                provider=self.provider,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except RateLimitError as e:
            return self._error_response(f"Límite de cuota excedido: {e}")
        except APIConnectionError as e:
            return self._error_response(f"Error de conexión: {e}")
        except APIError as e:
            return self._error_response(f"Error de la API de OpenAI: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except (RateLimitError, APIConnectionError, APIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
