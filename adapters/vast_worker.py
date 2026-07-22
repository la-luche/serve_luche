"""Backward-compatible Vast entrypoint for the portable HTTP worker."""
try:
    from adapters.http_worker import app, main
except ModuleNotFoundError:  # `python adapters/vast_worker.py` without PYTHONPATH
    from http_worker import app, main


if __name__ == "__main__":
    main()
