import torch
from torch import Tensor

def IoU_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU score for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = torch.clamp(input + target, min=0, max=1).sum(dim=sum_dim)

    # If both input and target is empty than it is an exact match 
    union = torch.where(union == 0, inter, union)

    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean()

if __name__ == '__main__':
    # Left side in the target is filled with ones
    target = torch.zeros((100, 100))
    target[:, :50] = 1 

    # Top half of the input is filled with ones
    input = torch.zeros((100, 100))
    input[:50, :] = 1

    iou = IoU_score(input, target)

    intersection = 50*50
    union = 50*100 + 50*100 - 50*50
    assert iou == (intersection / union)

    print(f'IoU score: {iou}')
