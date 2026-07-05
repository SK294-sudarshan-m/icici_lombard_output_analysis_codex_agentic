"""Full-artifact, project-model-driven claim case-verdict analyzer."""


def main(argv=None):
    """Lazily dispatch to the standalone CLI without importing it at package load."""
    from .agent import main as _main

    return _main(argv)

__all__ = ["main"]
