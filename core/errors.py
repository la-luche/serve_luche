class InferenceCancelled(RuntimeError):
    """Raised when a dedicated-worker job is cancelled between GPU batches."""
