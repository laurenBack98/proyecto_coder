"""Demo del cliente de LLM asíncrono.

Uso:
    pip install -r requirements.txt
    cp .env.example .env     # Completar las API keys
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
    ChatMessage(role="user", content="¿Cuántas provincias tiene Argentina?"),
]

# Proveedores obligatorios y adicionales para probar
PROVEEDORES = [
    (Provider.OPENAI, "gpt-4o-mini", "OPENAI_API_KEY", "openai_api_key"),
    (Provider.ANTHROPIC, "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY", "anthropic_api_key"),
    (Provider.GEMINI, "gemini-1.5-flash", "GOOGLE_API_KEY", "google_api_key"),
    (Provider.HUGGINGFACE, "deepseek-ai/DeepSeek-V4.1-Flash:novita", "HF_TOKEN", "huggingface_api_key"),
]


def probar_validacion() -> None:
    print("\n--- 1. Prueba de validación Pydantic ---")
    try:
        # Temperatura fuera de rango (debe fallar antes de tocar la API)
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5.0)
    except ValidationError as e:
        print("✅ Validación capturada con éxito antes de llamar a la API:\n", e)


async def probar_proveedor(provider: Provider, modelo: str, env_var: str, campo: str) -> None:
    print(f"\n--- Prueba Proveedor: {provider.value} ({modelo}) ---")
    key = os.environ.get(env_var)
    if not key:
        print(f"⚠️ Variable {env_var} no configurada. Se omite llamada real.")
        return

    config = LLMConfig(
        provider=provider,
        model=modelo,
        temperature=0.7,
        max_tokens=200,
        **{campo: SecretStr(key)},
    )
    manager = AsyncLLMManager(config)

    # 1. Modo respuesta completa
    print(f"[{provider.value}] Generación completa:")
    resultado = await manager.generate(PREGUNTA)
    if resultado.error:
        print(f"❌ Error controlado: {resultado.error}")
    else:
        print(f"Respuesta: {resultado.content}")

    # 2. Modo streaming
    print(f"[{provider.value}] Streaming:")
    async for chunk in manager.generate_stream(PREGUNTA):
        print(chunk, end="", flush=True)
    print()


async def probar_resiliencia() -> None:
    print("\n--- 2. Prueba de resiliencia ante error de API ---")
    config = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    resultado = await AsyncLLMManager(config).generate(PREGUNTA)
    print("¿El script continuó sin caerse (no crash)?: ✅ Sí")
    print("Mensaje de error controlado devuelto:", resultado.error)


async def main() -> None:
    load_api_keys()
    probar_validacion()
    
    for provider, modelo, env_var, campo in PROVEEDORES:
        await probar_proveedor(provider, modelo, env_var, campo)
        
    await probar_resiliencia()


if __name__ == "__main__":
    asyncio.run(main())
