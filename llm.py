"""Gemini LLM wrapper with retry/backoff."""

import time
import json
import config
from google import genai
from google.genai import types

_client = None


def get_client() -> genai.Client:
    """Get or create a Gemini client."""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def call_llm(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    response_mime_type: str | None = None,
    tools: list | None = None,
    system_instruction: str | None = None,
) -> str:
    """Call Gemini LLM with retry/backoff. Returns text response."""
    client = get_client()
    model = model or config.EXTRACTION_MODEL
    temperature = temperature if temperature is not None else config.EXTRACTION_TEMPERATURE

    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type=response_mime_type or "text/plain",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if system_instruction:
        gen_config.system_instruction = system_instruction
    if tools:
        gen_config.tools = tools

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            if response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            error_str = str(e)
            if attempt < config.MAX_RETRIES - 1:
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    delay = max(10.0, config.INITIAL_RETRY_DELAY * (config.RETRY_BACKOFF_FACTOR ** attempt))
                    print(f"  Rate limited (attempt {attempt + 1}/{config.MAX_RETRIES}), waiting {delay:.1f}s...")
                else:
                    delay = config.INITIAL_RETRY_DELAY * (config.RETRY_BACKOFF_FACTOR ** attempt)
                    print(f"  Error: {error_str[:100]}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"  Final LLM attempt failed: {error_str[:100]}")
                raise

    return ""


def call_llm_json(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    system_instruction: str | None = None,
) -> dict | list:
    """Call Gemini and parse JSON response."""
    text = call_llm(
        prompt=prompt,
        model=model,
        temperature=temperature,
        response_mime_type="application/json",
        system_instruction=system_instruction,
    )
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse JSON response: {text[:200]}")
            return []


def call_llm_with_tools(
    prompt: str,
    tool_declarations: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    system_instruction: str | None = None,
) -> dict:
    """Call Gemini with function-calling tools. Returns {tool_name, tool_args, reasoning}."""
    client = get_client()
    model = model or config.EXTRACTION_MODEL
    temperature = temperature if temperature is not None else config.EXTRACTION_TEMPERATURE

    # Convert tool declarations to Gemini format
    gemini_tools = []
    for tool in tool_declarations:
        properties = {}
        required = []
        for param_name, param_info in tool.get("parameters", {}).items():
            prop = {"type": param_info.get("type", "string")}
            if "description" in param_info:
                prop["description"] = param_info["description"]
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            properties[param_name] = prop
            if param_info.get("required", True):
                required.append(param_name)

        func_decl = types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties={k: types.Schema(**v) for k, v in properties.items()},
                required=required if required else None,
            ),
        )
        gemini_tools.append(func_decl)

    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        tools=[types.Tool(function_declarations=gemini_tools)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if system_instruction:
        gen_config.system_instruction = system_instruction

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )

            # Extract function call from response safely without triggering AFC or text part warnings
            if response.candidates and response.candidates[0].content.parts:
                text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
                reasoning = "".join(text_parts).strip()
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        return {
                            "tool_name": part.function_call.name,
                            "tool_args": dict(part.function_call.args) if part.function_call.args else {},
                            "reasoning": reasoning,
                        }
                # No function call — model responded with text instead
                return {
                    "tool_name": None,
                    "tool_args": {},
                    "reasoning": reasoning,
                }

            return {"tool_name": None, "tool_args": {}, "reasoning": ""}

        except Exception as e:
            error_str = str(e)
            if attempt < config.MAX_RETRIES - 1:
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    delay = max(10.0, config.INITIAL_RETRY_DELAY * (config.RETRY_BACKOFF_FACTOR ** attempt))
                    print(f"  Tool-calling rate limited (attempt {attempt + 1}/{config.MAX_RETRIES}), waiting {delay:.1f}s...")
                else:
                    delay = config.INITIAL_RETRY_DELAY * (config.RETRY_BACKOFF_FACTOR ** attempt)
                    print(f"  Tool-calling error: {error_str[:100]}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise

    return {"tool_name": None, "tool_args": {}, "reasoning": ""}


def generate_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    client = get_client()
    model = model or config.EMBEDDING_MODEL
    chunk_size = config.EMBEDDING_BATCH_SIZE

    embeddings = []
    # Batch in chunks
    for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + config.EMBEDDING_BATCH_SIZE]

        contents_payload = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        for attempt in range(10):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=contents_payload,
                )
                for emb in response.embeddings:
                    embeddings.append(emb.values)
                break
            except Exception as e:
                error_str = str(e)
                if attempt < 9:
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        delay = max(10.0, 5.0 * (1.5 ** attempt))
                        print(f"  Embedding rate limit hit (attempt {attempt + 1}/10), waiting {delay:.1f}s for quota refresh...")
                    else:
                        delay = 2.0 * (1.5 ** attempt)
                        print(f"  Embedding error ({error_str[:80]}), waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

        if i + chunk_size < len(texts):
            time.sleep(2.0)  # Brief delay between batches

    return embeddings
