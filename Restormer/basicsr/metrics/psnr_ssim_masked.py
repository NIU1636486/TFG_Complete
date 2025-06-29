import numpy as np
import torch


def _prepare_image(img):
    """Convert tensor to numpy in HWC and float64 format."""
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # CHW to HWC
    return img.astype(np.float64)


def _get_mask(target, lq):
    """Create a binary mask where target and lq differ."""
    diff = torch.abs(target - lq)
    mask = (diff != 0).float()
    return mask


def calculate_psnr_masked_clean(pred, target, lq, input_order='HWC'):
    """
    PSNR on clean regions (where mask == 0).

    Args:
        pred: predicted image (ndarray or tensor)
        target: ground truth image (ndarray or tensor)
        lq: low quality input used to compute mask
        input_order: 'HWC' or 'CHW'
    Returns:
        float: mean PSNR over clean pixels
    """
    pred = _prepare_image(pred)
    target = _prepare_image(target)
    lq = _prepare_image(lq)

    if input_order == 'CHW':
        pred = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
        lq = lq.transpose(1, 2, 0)

    pred_t = torch.tensor(pred).permute(2, 0, 1)
    target_t = torch.tensor(target).permute(2, 0, 1)
    lq_t = torch.tensor(lq).permute(2, 0, 1)

    mask = _get_mask(target_t, lq_t)  # shape: C x H x W
    clean_mask = (mask == 0).float()

    diff_sq = (pred_t - target_t) ** 2
    mse = (diff_sq * clean_mask).sum() / clean_mask.sum().clamp(min=1.0)

    max_val = 1.0 if pred.max() <= 1.0 else 255.0
    psnr = 20. * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_psnr_masked_dirty(pred, target, lq, input_order='HWC'):
    """
    PSNR on dirty regions (where mask == 1).

    Args:
        pred: predicted image (ndarray or tensor)
        target: ground truth image (ndarray or tensor)
        lq: low quality input used to compute mask
        input_order: 'HWC' or 'CHW'
    Returns:
        float: mean PSNR over dirty pixels
    """
    pred = _prepare_image(pred)
    target = _prepare_image(target)
    lq = _prepare_image(lq)

    if input_order == 'CHW':
        pred = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
        lq = lq.transpose(1, 2, 0)

    pred_t = torch.tensor(pred).permute(2, 0, 1)
    target_t = torch.tensor(target).permute(2, 0, 1)
    lq_t = torch.tensor(lq).permute(2, 0, 1)

    mask = _get_mask(target_t, lq_t)
    dirty_mask = (mask != 0).float()

    diff_sq = (pred_t - target_t) ** 2
    mse = (diff_sq * dirty_mask).sum() / dirty_mask.sum().clamp(min=1.0)

    max_val = 1.0 if pred.max() <= 1.0 else 255.0
    psnr = 20. * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()
