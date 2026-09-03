"""Package-specific exceptions."""


class GeometricCoderError(Exception):
    """Base exception for GeCo."""


class ConfigurationError(GeometricCoderError):
    """Raised when a project or geometry configuration is invalid."""


class DependencyCycleError(ConfigurationError):
    """Raised when geometry dependencies form a cycle."""


class MissingOptionalDependencyError(GeometricCoderError):
    """Raised when an optional feature dependency is unavailable."""


class ProjectExistsError(GeometricCoderError):
    """Raised when project creation would overwrite an existing project."""


class ProjectNotFoundError(GeometricCoderError):
    """Raised when a GeCo project cannot be opened."""
