import torch
import dataclasses
from torch import nn


@dataclasses.dataclass
class BayesianPRMConfig:
    n_classes: int
    hidden_dims: list[int]
    backbone_model_name: str
    freeze_backbone: bool = True


class BayesianPRM(nn.Module):
    def __init__(self, backbone: nn.Module, config: BayesianPRMConfig, device=None):
        super().__init__()
        self.backbone = backbone
        self.n_classes = config.n_classes
        self.device = device

        layers = []
        embedding_dim = self.backbone.config.hidden_size
        dims = [embedding_dim] + config.hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-1], self.n_classes))
        self.head = nn.Sequential(*layers)
        self.head.to(backbone.dtype)

        # freeze all backbone modules
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                if p.requires_grad:
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, emb_size)
        if attention_mask is not None:
            # Get last non-padded token hidden state for each seq. in batch
            last_token_indices = attention_mask.sum(dim=-1) - 1
            batch_indices = torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device
            )
            pooled_output = last_hidden_state[batch_indices, last_token_indices]
        else:
            pooled_output = last_hidden_state[:, -1]
        return self.head(pooled_output)


    def mean(self, logits) -> torch.Tensor:
        pmf = logits.softmax(dim=-1)
        x = torch.arange(logits.shape[-1]) / logits.shape[-1] 
        x = x.to(logits)
        return pmf @ x


    def variance(self, logits) -> torch.Tensor:
        pmf = logits.softmax(dim=-1)
        x = torch.arange(logits.shape[-1]).to(logits)
        x = x.to(logits)
        x_square = x**2
        mean = pmf @ x
        return pmf @ x_square - mean**2


    def ucb(self, logits, beta=1) -> torch.Tensor:
        pmf = logits.softmax(dim=-1)
        x = torch.arange(logits.shape[-1]) / logits.shape[-1] 
        x = x.to(logits)
        x_square = x**2
        mean = pmf @ x
        var = pmf @ x_square - mean**2
        return mean + beta * var