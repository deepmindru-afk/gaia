"""Walk a response annotation for the parts that generate ``unknown``.

``Any``, a bare ``dict``/``list`` and a ``dict[str, Any]`` value all export as
an unconstrained schema, which openapi-typescript renders as ``unknown`` — so
every consumer casts. The walk is recursive because the annotation on the route
is only the outermost layer: a perfectly typed ``GmailThreadResponse`` can still
carry a ``dict[str, Any]`` field, and that field is what the client has to cast.
The models it passes through are reported too — a serializer option set on the
route applies to all of them, not just the outermost one.
"""

import dataclasses
import types
import typing
from typing import Any

from pydantic import BaseModel

_UNPARAMETRIZED = (dict, list, set, tuple, frozenset)
_SEQUENCE_ORIGINS = (list, set, tuple, frozenset)


@dataclasses.dataclass
class ResponseWalk:
    """What one annotation reaches: loose ``Model.field`` paths, and every model."""

    untyped: set[str] = dataclasses.field(default_factory=set)
    visited: set[type] = dataclasses.field(default_factory=set)

    @property
    def models(self) -> set[type[BaseModel]]:
        return {t for t in self.visited if issubclass(t, BaseModel)}


def _walk(annotation: Any, owner: str, out: ResponseWalk) -> None:
    if hasattr(annotation, "__metadata__"):  # Annotated[X, ...] -> X
        annotation = typing.get_args(annotation)[0]
    origin = typing.get_origin(annotation)

    if origin in _SEQUENCE_ORIGINS:
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        if not args:
            out.untyped.add(owner)
        for arg in args:
            _walk(arg, owner, out)
        return
    if origin is dict:
        key, value = typing.get_args(annotation)
        if key is not str:
            out.untyped.add(owner)
        _walk(value, owner, out)
        return
    if origin in (types.UnionType, typing.Union):
        for arg in typing.get_args(annotation):
            if arg is not type(None):
                _walk(arg, owner, out)
        return
    if annotation is Any or annotation is object or annotation in _UNPARAMETRIZED:
        out.untyped.add(owner)
        return

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in out.visited:
            return
        out.visited.add(annotation)
        for name, field in annotation.model_fields.items():
            _walk(field.annotation, f"{annotation.__name__}.{name}", out)
        return
    if dataclasses.is_dataclass(annotation) or typing.is_typeddict(annotation):
        if annotation in out.visited:
            return
        out.visited.add(annotation)
        for name, hint in typing.get_type_hints(annotation).items():
            _walk(hint, f"{annotation.__name__}.{name}", out)


def walk_response(annotation: Any, owner: str) -> ResponseWalk:
    """Everything ``annotation`` reaches, with ``owner`` naming the annotation itself.

    A nested model's own name owns every field below it, so one loose field is
    reported once however many routes reach it; ``owner`` shows up only when
    the route's own annotation is the loose one.
    """
    walk = ResponseWalk()
    _walk(annotation, owner, walk)
    return walk
