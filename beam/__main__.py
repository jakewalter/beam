"""Convenience entrypoint for BEAM.

This allows running the main pipeline via:

  python -m beam --config configs/template_config.json

It simply delegates to the existing script wrapper `scripts/run_pipeline.py`.
"""
import os
import sys


def main():
    # Ensure repository root is on sys.path so 'scripts' can be imported when
    # running via `python -m beam` regardless of current working directory.
    # repository root (one directory above the 'beam' package)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import and call the script's main() by loading the script file directly
    # (this avoids relying on package import mechanics for the `scripts` dir).
    try:
        import importlib.util

        script_path = os.path.join(repo_root, 'scripts', 'run_pipeline.py')
        spec = importlib.util.spec_from_file_location('run_pipeline', script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    except Exception as e:
        sys.stderr.write(f'Failed to execute pipeline runner from {script_path}: {e}\n')
        raise


if __name__ == '__main__':
    main()
