from typing import AsyncGenerator, List, Type

from llm_client.base import BaseLLMClient
from llm_client.clients.anthropic_client import AnthropicClient
from llm_client.clients.gemini_client import GeminiClient
from llm_client.clients.huggingface_client import HuggingFaceClient
from llm_client.clients.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, Provider

# Registro: proveedor -> (clase del cliente, nombre del campo con la API key en LLMConfig).
# Para agregar un proveedor nuevo, basta con sumar una línea acá.
_REGISTRO: dict[Provider, tuple[Type[BaseLLMClient], str]] = {
    Provider.OPENAI: (OpenAIClient, "openai_api_key"),
    Provider.ANTHROPIC: (AnthropicClient, "anthropic_api_key"),
    Provider.GEMINI: (GeminiClient, "google_api_key"),
    Provider.HUGGINGFACE: (HuggingFaceClient, "huggingface_api_key")
}


class AsyncLLMManager:
    """Factory: construye el cliente correcto según config.provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        if self.config.provider not in _REGISTRO:
            raise ValueError(f"Proveedor no soportado: {self.config.provider}")

        cliente_cls, campo_key = _REGISTRO[self.config.provider]
        api_key = getattr(self.config, campo_key)
        if not api_key:
            raise ValueError(f"Falta {campo_key} en la configuración")

        return cliente_cls(
            api_key=api_key.get_secret_value(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages):
            yield chunk
