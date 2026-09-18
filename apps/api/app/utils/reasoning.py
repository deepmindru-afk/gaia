"""Read a model's "thinking" off a streamed chunk.

Deliberately dependency-free (langchain only): both the comms stream and the
subagent runner need it, and agent_utils, the obvious-looking home, pulls in
the tool and subagent registries, which would make this a cycle.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk


def extract_reasoning_delta(chunk: AIMessage | AIMessageChunk) -> str:
    """Pull this chunk's reasoning ("thinking") text, model-agnostic.

    ChatOpenRouter surfaces reasoning as standard reasoning content blocks;
    other providers (DeepSeek-style) put it in additional_kwargs.reasoning_content.
    Returns "" when the chunk carries no thinking, so the caller emits nothing.
    """
    parts: list[str] = []
    for block in getattr(chunk, "content_blocks", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "reasoning":
            text = (
                block.get("reasoning")
                if isinstance(block, dict)
                else getattr(block, "reasoning", "")
            )
            if text:
                parts.append(text)
    if not parts:
        fallback = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
        if fallback:
            parts.append(fallback if isinstance(fallback, str) else str(fallback))
    return "".join(parts)
