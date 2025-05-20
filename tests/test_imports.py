"""Test that all imports work as expected."""


def test_import_adaptive_moe():
    """Test that the adaptive_moe package can be imported."""
    import adaptive_moe  # noqa: F401


def test_import_router():
    """Test that the router module can be imported."""
    from adaptive_moe.router import UncertaintyRouter  # noqa: F401


def test_import_utils():
    """Test that utility modules can be imported."""
    from adaptive_moe.utils.config import AdaptiveMoEConfig  # noqa: F401
    from adaptive_moe.utils.logging import setup_logging  # noqa: F401
    from adaptive_moe.utils.model_utils import load_base_model  # noqa: F401
