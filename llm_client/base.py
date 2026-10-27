from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple
from llm_client.schemas import ChatMessage, ModelResponse, Provider


class BaseLLMClient(ABC):
    """Contrato que todo cliente de LLM debe cumplir."""

    provider: Provider

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._build_sdk_client(api_key)

    @abstractmethod
    def _build_sdk_client(self, api_key: str):
        """Crea el cliente async del SDK del proveedor."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (streaming)."""
        raise NotImplementedError
        yield

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Alias compatible para streaming."""
        async for chunk in self.generate_stream(messages):
            yield chunk

    def _error_response(self, mensaje: str) -> ModelResponse:
        """Respuesta de error uniforme para no romper el event loop."""
        return ModelResponse(provider=self.provider, model=self.model, content="", error=mensaje)

    @staticmethod
    def _separar_system(messages: List[ChatMessage]) -> Tuple[Optional[str], List[ChatMessage]]:
        """Separa los mensajes system del resto."""
        system = [m.content for m in messages if m.role == "system"]
        resto = [m for m in messages if m.role != "system"]
        return ("\n\n".join(system) if system else None), resto
