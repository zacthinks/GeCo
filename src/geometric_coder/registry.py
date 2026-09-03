"""Geometry-spec normalization and dependency resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from geometric_coder.exceptions import ConfigurationError, DependencyCycleError
from geometric_coder.geometry.base import Geometry


@dataclass(frozen=True, slots=True)
class RegisteredGeometry:
    """A geometry with a stable project-local registry name."""

    name: str
    geometry: Geometry
    public: bool


class GeometryRegistry:
    """Normalize nested recipe bundles and resolve geometry dependencies."""

    def __init__(self) -> None:
        self._public_by_name: dict[str, Geometry] = {}
        self._public_name_by_object: dict[int, str] = {}
        self._hidden_name_by_object: dict[int, str] = {}

    @property
    def public(self) -> dict[str, Geometry]:
        """Return a copy of public geometry registrations."""
        return dict(self._public_by_name)

    def register_spec(self, spec: Mapping[str, Any]) -> None:
        """Register a top-level mapping of names to geometries or recipe bundles."""
        if not isinstance(spec, Mapping):
            raise TypeError("geometries must be a mapping from names to geometry specifications")
        for path, geometry in self._flatten(spec):
            name = "_".join(path)
            self._register_public(name, geometry)

    def _flatten(
        self,
        spec: Mapping[str, Any],
        prefix: tuple[str, ...] = (),
    ) -> Iterator[tuple[tuple[str, ...], Geometry]]:
        for raw_name, value in spec.items():
            name = str(raw_name).strip()
            if not name:
                raise ConfigurationError("Geometry names may not be empty.")
            path = (*prefix, name)
            if isinstance(value, Geometry):
                yield path, value
            elif isinstance(value, Mapping):
                yield from self._flatten(value, path)
            else:
                raise TypeError(
                    f"Geometry specification at {'/'.join(path)!r} must be a Geometry "
                    "or a mapping."
                )

    def _register_public(self, name: str, geometry: Geometry) -> None:
        if name in self._public_by_name:
            raise ConfigurationError(f"Duplicate normalized geometry name: {name!r}")
        object_id = id(geometry)
        existing_name = self._public_name_by_object.get(object_id)
        if existing_name is not None and existing_name != name:
            raise ConfigurationError(
                f"The same Geometry object was registered under both {existing_name!r} "
                f"and {name!r}. Create two geometry objects if two public entries are intended."
            )
        self._public_by_name[name] = geometry
        self._public_name_by_object[object_id] = name

    def _name_for(self, geometry: Geometry) -> str:
        object_id = id(geometry)
        public_name = self._public_name_by_object.get(object_id)
        if public_name is not None:
            return public_name
        hidden_name = self._hidden_name_by_object.get(object_id)
        if hidden_name is None:
            hidden_name = f"__dependency_{len(self._hidden_name_by_object) + 1}"
            self._hidden_name_by_object[object_id] = hidden_name
        return hidden_name

    def build_order(self) -> list[RegisteredGeometry]:
        """Return all dependencies in topological computation order."""
        visiting: set[int] = set()
        visited: set[int] = set()
        ordered: list[RegisteredGeometry] = []

        def visit(geometry: Geometry) -> None:
            object_id = id(geometry)
            if object_id in visited:
                return
            if object_id in visiting:
                raise DependencyCycleError(
                    f"Geometry dependency cycle detected at {self._name_for(geometry)!r}."
                )
            visiting.add(object_id)
            for dependency in geometry.dependencies:
                if not isinstance(dependency, Geometry):
                    raise ConfigurationError(
                        f"{type(geometry).__name__} contains a non-Geometry dependency."
                    )
                visit(dependency)
            visiting.remove(object_id)
            visited.add(object_id)
            name = self._name_for(geometry)
            ordered.append(
                RegisteredGeometry(
                    name=name,
                    geometry=geometry,
                    public=object_id in self._public_name_by_object,
                )
            )

        for geometry in self._public_by_name.values():
            visit(geometry)
        return ordered
