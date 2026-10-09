# Proyecto_Coder
# Cliente LLM Asíncrono

Este proyecto es una pequeña base para interactuar con distintos proveedores de modelos de lenguaje (LLM) desde una sola interfaz en Python. Está pensado como una demo de integración y comparación de respuestas entre OpenAI, Anthropic, Gemini y Hugging Face, con soporte para llamadas asíncronas y streaming.

## ¿Qué hace?

El proyecto centraliza la configuración, la validación y la creación del cliente correcto según el proveedor elegido. A partir de una configuración simple, se puede generar una respuesta o stream de texto sin tener que reescribir lógica para cada API.

Incluye:

- Soporte para varios proveedores: OpenAI, Anthropic, Gemini y Hugging Face
- Validación de configuración con Pydantic
- Manejo seguro de API keys con variables de entorno y `.env`
- Llamadas asíncronas con `asyncio`
- Streaming de respuesta en tiempo real
- Tratamiento de errores para que la ejecución no termine abruptamente

## Estructura del proyecto

```text
proyecto/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── llm_client/
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   ├── manager.py
│   ├── schemas.py
│   └── clients/
│       ├── __init__.py
│       ├── openai_client.py
│       ├── anthropic_client.py
│       ├── gemini_client.py
│       └── huggingface_client.py
└── README.md
```

## Tecnologías utilizadas

- Python 3
- `asyncio`
- `pydantic`
- `python-dotenv`
- clientes de distintas APIs de LLM

## Requisitos

- Python 3.10 o superior
- Acceso a una o más claves API de proveedores compatibles
- Dependencias instaladas desde `requirements.txt`

## Instalación

1. Cloná el proyecto.
2. Crear un entorno virtual:

```bash
python -m venv .venv
```

3. Activá el entorno virtual:

- En Windows:

```bash
.venv\Scripts\activate
```

- En macOS/Linux:

```bash
source .venv/bin/activate
```

4. Instalá las dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

El proyecto lee las API keys desde variables de entorno o desde un archivo `.env`.

Ejemplo de `.env`:

```env
OPENAI_API_KEY=tu_key_openai
ANTHROPIC_API_KEY=tu_key_anthropic
GOOGLE_API_KEY=tu_key_google
HF_TOKEN=tu_token_huggingface
```

También puede cargar claves interactivamente desde la consola si se configura la opción de modo interactivo.

## Ejecución

Para probar el cliente con los proveedores configurados:

```bash
python main.py
```

El script actual crea una pregunta de ejemplo y luego intenta consultar varios proveedores. Si falta alguna clave, lo omite y sigue con el siguiente. Si una clave es inválida, captura el error sin romper la ejecución.

## Cómo funciona internamente

El flujo principal es:

1. Se cargan las variables de entorno.
2. Se valida la configuración con `LLMConfig`.
3. Se crea un `AsyncLLMManager` según el proveedor.
4. Se invoca `generate()` o `generate_stream()`.
5. Se devuelve un `ModelResponse` con el contenido o un mensaje de error.

## Validación y resiliencia

El proyecto incluye una validación previa para detectar configuraciones incorrectas antes de llamar a la API. También prueba un caso de error con una clave inválida para comprobar que la aplicación sigue funcionando y que el error se maneja de forma segura.

## Casos de uso

Este tipo de estructura sirve para:

- Comparar respuestas entre distintos modelos
- Crear prototipos de chatbots o asistentes
- Probar streaming de contenido en tiempo real
- Construir una base reutilizable para integrar APIs LLM en proyectos más grandes

## Ideas de mejora

- Agregar más proveedores
- Soportar historial completo de conversación
- Añadir logging y trazabilidad
- Crear una API REST para consumir el cliente desde frontend o servicios
- Estructurar el proyecto en paquetes más complejos con tests

## Autor

Proyecto personal en desarrollo, orientado a explorar la integración de múltiples modelos de lenguaje con una interfaz unificada.
