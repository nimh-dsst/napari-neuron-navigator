"""Live, named neuron cohorts defined by annotation rules and keyed by file_id."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from .neuron_table_ops import annotation_matches


@dataclass(frozen=True)
class AnnotationRule:
    label_filter: str | None = None
    tag_filter: tuple[str, ...] | str | None = None
    tag_match: str = "all"
    tag_count: str = "any"
    excluded_tags: tuple[str, ...] = ()

    def __post_init__(self):
        tags = self.tag_filter
        if tags is not None and tags != "":
            tags = (tags,) if isinstance(tags, str) else tags
            object.__setattr__(self, "tag_filter", tuple(sorted(set(tags))))
        object.__setattr__(
            self, "excluded_tags", tuple(sorted(set(self.excluded_tags)))
        )
        # Validate modes even when the rule currently has no matches.
        annotation_matches("", (), **asdict(self))

    def members(self, entries: Mapping) -> tuple[str, ...]:
        """Evaluate against the entire Data cohort, independently of row filters."""
        rule = asdict(self)
        return tuple(
            sorted(
                str(file_id)
                for file_id, entry in entries.items()
                if annotation_matches(entry.label, entry.tags, **rule)
            )
        )

    def description(self) -> str:
        parts = []
        if self.label_filter is not None:
            parts.append(f"Label = {self.label_filter!r}")
        if self.tag_filter == "":
            parts.append("No tags")
        elif self.tag_filter is not None:
            parts.append(f"{self.tag_match}: {', '.join(self.tag_filter)}")
        if self.excluded_tags:
            parts.append(f"exclude: {', '.join(self.excluded_tags)}")
        if self.tag_count != "any":
            parts.append(f"tag count: {self.tag_count}")
        return "; ".join(parts) or "All neurons in Data"

    def to_state(self):
        state = asdict(self)
        if isinstance(self.tag_filter, tuple):
            state["tag_filter"] = list(self.tag_filter)
        state["excluded_tags"] = list(self.excluded_tags)
        return state

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, Mapping):
            raise TypeError("A neuron set needs an annotation rule.")
        return cls(
            **{key: state[key] for key in cls.__dataclass_fields__ if key in state}
        )


@dataclass(frozen=True)
class NeuronSet:
    name: str
    rule: AnnotationRule
    set_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        if not self.name.strip() or not self.set_id.strip():
            raise ValueError("Neuron sets need a nonempty name and identifier.")
        object.__setattr__(self, "name", self.name.strip())

    def to_state(self):
        return {"name": self.name, "rule": self.rule.to_state(), "set_id": self.set_id}


class NeuronSetStore:
    """Definitions persist; memberships are always evaluated from current rows."""

    def __init__(self):
        self._sets: dict[str, NeuronSet] = {}

    def sets(self):
        return tuple(self._sets.values())

    def get(self, set_id):
        return self._sets.get(set_id)

    def save(self, name, rule, *, set_id=None):
        item = NeuronSet(name=name, rule=rule, **({"set_id": set_id} if set_id else {}))
        if any(
            s.name.casefold() == item.name.casefold() and s.set_id != item.set_id
            for s in self._sets.values()
        ):
            raise ValueError("A neuron set with that name already exists.")
        self._sets[item.set_id] = item
        return item

    def remove(self, set_id):
        self._sets.pop(set_id, None)

    def to_state(self):
        return {"version": 1, "sets": [item.to_state() for item in self.sets()]}

    def load_state(self, state, *, merge=False):
        incoming = NeuronSetStore()
        for raw in state.get("sets", []):
            incoming.save(
                raw["name"],
                AnnotationRule.from_state(raw["rule"]),
                set_id=raw["set_id"],
            )
        if merge:
            for item in incoming.sets():
                if self.get(item.set_id) is None:
                    self.save(item.name, item.rule, set_id=item.set_id)
        else:
            self._sets = incoming._sets


def membership_overlap(a, b):
    """Count overlap using full file_id identity, including zero-size sets."""
    left, right = set(map(str, a)), set(map(str, b))
    shared = len(left & right)
    union = len(left | right)
    return {
        "a": len(left),
        "b": len(right),
        "intersection": shared,
        "union": union,
        "a_only": len(left - right),
        "b_only": len(right - left),
        "jaccard": shared / union if union else None,
        "dice": 2 * shared / (len(left) + len(right)) if left or right else None,
        "fraction_a": shared / len(left) if left else None,
        "fraction_b": shared / len(right) if right else None,
    }
