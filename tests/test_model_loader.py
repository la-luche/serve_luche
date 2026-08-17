import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from core import model_loader


class _Registry:
    def build(self, config):
        if config == {"kind": "preprocessor"}:
            return object()
        return torch.nn.Linear(2, 2)


class MetaModelLoaderTests(unittest.TestCase):
    def test_meta_loader_assigns_real_checkpoint_tensors(self):
        config = SimpleNamespace(
            model={"backbone": {}, "kind": "model"},
            data_preprocessor={"kind": "preprocessor"},
            test_pipeline=[],
        )
        state = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([5.0, 6.0]),
        }

        with patch(
            "sapiens.engine.config.Config.fromfile", return_value=config
        ), patch("sapiens.registry.MODELS", _Registry()), patch(
            "safetensors.torch.load_file", return_value=state
        ):
            model = model_loader._load_safetensors_meta(
                "config.py", "checkpoint.safetensors", "cpu"
            )

        self.assertFalse(any(value.is_meta for value in model.parameters()))
        torch.testing.assert_close(model.weight, state["weight"])
        torch.testing.assert_close(model.bias, state["bias"])


if __name__ == "__main__":
    unittest.main()
