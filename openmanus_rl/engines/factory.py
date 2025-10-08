"""Engine factory helpers.

Exposes `create_llm_engine` returning a callable that maps prompt -> text using
the minimal `ChatOpenAI` wrapper. Keep the surface small and stable so tools
can depend on it without heavy coupling.
"""

from typing import Any, Callable, Optional
from .openai import ChatOpenAI


def create_llm_engine(
    model_string: str = "gpt-4o-mini",
    is_multimodal: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Callable[[Any], Any]:
    chat = ChatOpenAI(
        model=model_string,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )

    def _engine(prompt: Any, *args: Any, **kwargs: Any) -> Any:
        # Tools currently call engine(prompt) for both text-only and multimodal
        # flows. Preserve backward compatibility by forwarding all additional
        # arguments directly to the underlying ChatOpenAI instance.
        return chat(prompt, *args, **kwargs)

    return _engine
