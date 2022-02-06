import torch.nn as nn


class ModelSessionManager:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __enter__(self) -> nn.Module:
        self.model.train()

        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model.eval()
