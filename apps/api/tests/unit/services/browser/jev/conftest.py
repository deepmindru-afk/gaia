"""Shared fakes: a Browser-Use-shaped DOM node and state summary, no browser needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


@dataclass
class FakeAXProperty:
    name: str
    value: object


@dataclass
class FakeAXNode:
    role: str | None = None
    name: str | None = None
    properties: list[FakeAXProperty] = field(default_factory=list)


@dataclass
class FakeRect:
    """``DOMRect`` in document coordinates, as ``absolute_position`` carries it."""

    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 20.0


@dataclass
class FakeNode:
    """The slice of ``EnhancedDOMTreeNode`` the observation reads."""

    node_name: str
    attributes: dict[str, str] = field(default_factory=dict)
    text: str = ""
    ax_node: FakeAXNode | None = None
    children_nodes: list[FakeNode] = field(default_factory=list)
    is_visible: bool | None = None
    absolute_position: FakeRect | None = None

    def get_meaningful_text_for_llm(self) -> str:
        for attr in ("value", "aria-label", "title", "placeholder", "alt"):
            if self.attributes.get(attr):
                return self.attributes[attr]
        return self.get_all_children_text()

    def get_all_children_text(self) -> str:
        return " ".join(
            [self.text, *(c.get_all_children_text() for c in self.children_nodes)]
        ).strip()


def make_page_info(
    *, viewport_height: int = 800, viewport_width: int = 1000, scroll_y: int = 0, scroll_x: int = 0
):
    return SimpleNamespace(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        page_width=viewport_width,
        page_height=100000,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
        pixels_above=scroll_y,
        pixels_below=0,
        pixels_left=0,
        pixels_right=0,
    )


def make_state(
    selector_map: dict[int, FakeNode],
    *,
    url: str = "https://x",
    title: str = "X",
    page_info=None,
):
    text = "\n".join(
        f"[{i}]<{n.node_name.lower()}>{n.get_all_children_text()}" for i, n in selector_map.items()
    )
    dom_state = SimpleNamespace(selector_map=selector_map, llm_representation=lambda: text)
    return SimpleNamespace(dom_state=dom_state, url=url, title=title, page_info=page_info)


@pytest.fixture
def flights_state():
    """Return a tiny Google-Flights-like page: two comboboxes, a native select, a button."""
    return make_state(
        {
            17: FakeNode(
                "INPUT", {"role": "combobox", "placeholder": "Where from?", "value": "Zurich"}
            ),
            23: FakeNode("INPUT", {"role": "combobox", "placeholder": "Where to?"}),
            31: FakeNode(
                "SELECT",
                {"value": "economy"},
                ax_node=FakeAXNode(role="combobox", name="Cabin class"),
                children_nodes=[
                    FakeNode("OPTION", {"value": "economy"}, text="Economy"),
                    FakeNode("OPTION", {"value": "business"}, text="Business"),
                    FakeNode("OPTION", {"value": "first", "disabled": ""}, text="First"),
                ],
            ),
            40: FakeNode("BUTTON", text="Search", ax_node=FakeAXNode(role="button", name="Search")),
            41: FakeNode(
                "INPUT",
                {"type": "checkbox", "aria-label": "Nonstop only", "checked": ""},
                ax_node=FakeAXNode(role="checkbox", properties=[FakeAXProperty("checked", True)]),
            ),
        }
    )
