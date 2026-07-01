"""Platform-agnostic Sapiens 2B keypoint inference core.

Nothing in this package imports RunPod, Vast, or any platform SDK. The serverless
adapters (adapters/) are thin shims that call `core.infer.run_video`.
"""
