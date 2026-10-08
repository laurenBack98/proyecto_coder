from typing import AsyncGenerator, List

from google import genai
from google.genai import types

from llm_client.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelResponse, Provider


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI

    def _build_sdk_client(self, api_key: str) -> genai.Client:
        return genai.Client(api_key=api_key)

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""
        system_instruction, resto = self._separar_system(messages)
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in resto
        ]
        return contents, system_instruction

    def _config(self, system_instruction) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            return ModelResponse(provider=self.provider, model=self.model, content=response.text or "")
        except Exception as e:
            return self._error_response(f"Error de la API de Gemini: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
