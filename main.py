"""Demo del cliente de LLM asíncrono.

Uso (desde esta carpeta):
    pip install -r requirements.txt
    cp .env.example .env     # y completá tus keys
    python main.py
"""
import asyncio
import os

from pydantic import SecretStr, ValidationError

from llm_client.config import load_api_keys
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, LLMConfig, Provider

PREGUNTA = [
    ChatMessage(role="system", content="Sos un profesor claro y conciso."),
    ChatMessage(role="user", content="cuantas provincias tiene argentina?"),
]

# (proveedor, modelo, variable de entorno, campo de LLMConfig)
PROVEEDORES = [
    (Provider.OPENAI, "gpt-4o-mini", "OPENAI_API_KEY", "openai_api_key"),
    (Provider.ANTHROPIC, "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY", "anthropic_api_key"),
    (Provider.GEMINI, "gemini-flash-latest", "GOOGLE_API_KEY", "google_api_key"),
    (Provider.HUGGINGFACE, "deepseek-ai/DeepSeek-V4.1-Flash:novita", "HF_TOKEN", "huggingface_api_key"),
]


def probar_validacion() -> None:
    try:
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)
    except ValidationError as e:
        print("❌ Se detectó ANTES de llamar a la API:\n", e)


async def probar_proveedor(provider, modelo, env_var, campo) -> None:
    key = os.environ.get(env_var)
    if not key:
        print(f"⏭️  {provider.value}: sin {env_var}, se omite.")
        return

    config = LLMConfig(
        provider=provider, model=modelo, temperature=0.7, max_tokens=200,
        **{campo: SecretStr(key)},
    )
    manager = AsyncLLMManager(config)

    resultado = await manager.generate(PREGUNTA)
    print(f"\n[{provider.value}]", resultado.content if not resultado.error else f"❌ {resultado.error}")

    print(f"[{provider.value}] streaming: ", end="")
    async for chunk in manager.generate_stream(PREGUNTA):
        print(chunk, end="", flush=True)
    print()


async def probar_resiliencia() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI, model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    resultado = await AsyncLLMManager(config).generate(PREGUNTA)
    print("\n¿El programa siguió vivo?: ✅ Sí")
    print("Error capturado (sin crash):", resultado.error)


async def main() -> None:
    load_api_keys()  
    probar_validacion()
    for p in PROVEEDORES:
        await probar_proveedor(*p)
    await probar_resiliencia()


if __name__ == "__main__":
    asyncio.run(main())
