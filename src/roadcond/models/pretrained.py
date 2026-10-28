"""Pretrained backbones from timm (PRD C4): new linear head, freeze or unfreeze the rest."""

import timm
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def build_pretrained(
    timm_name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """timm model (e.g. resnet50, efficientnet_b0) whose classifier is replaced by a new head."""
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    if freeze_backbone:
        set_backbone_trainable(model, False)
    return model


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(classifier head parameters, all other parameters)."""
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    return head, [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the head, which always trains."""
    head, backbone = split_parameters(model)
    for p in backbone:
        p.requires_grad = trainable
    for p in head:
        p.requires_grad = True


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Put BatchNorm layers in eval mode so a frozen backbone keeps its running statistics.

    Call after model.train(), since train() switches every layer back to training mode.
    """
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
