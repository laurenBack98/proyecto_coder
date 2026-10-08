from typing import AsyncGenerator, List

from huggingface_hub import AsyncInferenceClient

from llm_client.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelResponse, Provider


class HuggingFaceClient(BaseLLMClient):
    provider = Provider.HUGGINGFACE

    def _build_sdk_client(self, api_key: str) -> AsyncInferenceClient:
        return AsyncInferenceClient(token=api_key)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content or ""
            return ModelResponse(
                provider=self.provider,
                model=self.model,
                content=content,
            )
        except Exception as e:
            return self._error_response(f"Error de la API de HuggingFace: {e}")

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
                delta = chunk.choices[0].delta.content if getattr(chunk, "choices", None) else None
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"