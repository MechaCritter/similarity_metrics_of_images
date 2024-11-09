import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["MultiClassDiceLoss"]

def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class MultiClassDiceLoss(_Loss):
    __name__ = "MultiClassDiceLoss"
    def __init__(
                self,
                mode: str,
                classes: Optional[torch.Tensor] = None,
                log_loss: bool = False,
                from_logits: bool = True,
                smooth: float = 0.0,
                ignore_index: Optional[int] = None,
                eps: float = 1e-7,
            ):
        """
        Dice loss for image segmentation task.
        """
        super().__init__()
        assert mode in {'binary', 'multiclass'}, f"Unknown mode: {mode}. Supported modes are 'multiclass' and 'binary'."
        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param y_pred: (B, C, H, W)
        :param y_true: (B, C, H, W)

        :return: dice loss
        """
        assert y_true.size() == y_pred.size(), f"'y_true' and 'y_pred' must have the same shape, got {y_true.size()} for y_true and {y_pred.size()} for y_pred"
        logging.debug(f"""
                    Batch size: {y_true.size(0)}
                    Number of classes: {y_true.size(1)}
                    Image shape: ({y_true.size(2)} x {y_true.size(3)})
                    """)
        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            elif self.mode == 'binary':
                y_pred = F.logsigmoid(y_pred).exp()

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(batch_size, num_classes, -1)
        y_pred = y_pred.view(batch_size, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0 # Zeros out loss of classes that are not present in the mask (otherwise they would have a loss of 1, which is nonsense!
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


if __name__ == "__main__":
    from src.datasets import ExcavatorDataset
    from models.Segmentation import DeepLabV3Model
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    trainloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='train',
                                              return_type='image+mask',
                                              one_hot_encode_mask=True
                                              ), batch_size=10, shuffle=False) # TODO: setting batch size above 1 won't work?
    validloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='validation',
                                              return_type='image+mask',
                                                one_hot_encode_mask=True
                                              ), batch_size=1, shuffle=False)
    model = DeepLabV3Model(model_path='models/torch_model_files/DeepLabV3.pt').model
    criterion = MultiClassDiceLoss(mode='multiclass', ignore_index=0)
    for i, (images, masks) in enumerate(trainloader):
        images = images.to('cuda')
        masks = masks.to('cuda')
        output = model(images)
        loss = criterion(output, masks)
        print(loss)
        if i == 1:
            break
