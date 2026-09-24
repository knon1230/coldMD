"""Cold-compression molecular dynamics with Python-importable potentials."""

# Packaging metadata declares the identical release value in ``pyproject.toml``.
__version__ = "2.1.1"

from .config import (  # noqa: E402
    ColdMDConfig,
    ConfigurationError,
    RootConfig,
    SimulationConfig,
    load_config,
    load_config_dict,
)

__all__ = [
    "ColdMDConfig",
    "ConfigurationError",
    "RootConfig",
    "SimulationConfig",
    "__version__",
    "load_config",
    "load_config_dict",
]
