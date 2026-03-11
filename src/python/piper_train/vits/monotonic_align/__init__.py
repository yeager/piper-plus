import logging

import numpy as np
import torch


_logger = logging.getLogger(__name__)

try:
    from .core import maximum_path_c

    _has_cython = True
except ImportError:
    maximum_path_c = None
    _has_cython = False

try:
    from ._mas_jit import mas_backward, mas_forward

    _has_jit = True
except Exception:
    _has_jit = False

# Memory threshold for adaptive chunking (t_t * t_s)
# When matrix size exceeds this, use batch_size=1
_LARGE_MATRIX_THRESHOLD = 500000  # ~700x700


@torch.no_grad()
def maximum_path_pytorch(neg_cent, mask):
    """Pure PyTorch implementation of Monotonic Alignment Search.

    Runs entirely on the input tensor's device (GPU or CPU) without
    any CPU-GPU data transfers.  Faithfully reproduces the Cython
    ``maximum_path_each`` / ``maximum_path_c`` algorithm.

    When TorchScript is available the hot loops are JIT-compiled,
    giving ~6x speedup over the Cython+CPU-transfer path on GPU.

    Dimensions follow VITS convention:
        - neg_cent : [batch, t_y, t_x]   (t_y = audio frames, t_x = text tokens)
        - mask     : [batch, t_y, t_x]

    Returns:
        path : [batch, t_y, t_x]  binary alignment (same dtype as neg_cent)
    """
    dtype = neg_cent.dtype

    # Per-sample actual lengths from mask
    t_y_lens = mask.sum(1)[:, 0].long()  # [b]
    t_x_lens = mask.sum(2)[:, 0].long()  # [b]

    # Work in float32 for numerical stability; clone because forward
    # pass accumulates in-place (same semantics as Cython).
    value = neg_cent.clone().float()

    if _has_jit:
        value = mas_forward(value, t_y_lens, t_x_lens)
        path = mas_backward(value, t_y_lens, t_x_lens, dtype)
    else:
        value = _mas_forward_eager(value, t_y_lens, t_x_lens)
        path = _mas_backward_eager(value, t_y_lens, t_x_lens, dtype)

    return path * mask.to(dtype)


def _mas_forward_eager(value, t_y_lens, t_x_lens):
    """Forward DP pass — eager (non-JIT) fallback."""
    device = value.device
    b, t_y_max, t_x_max = value.shape

    max_neg = torch.tensor(-1e9, device=device, dtype=torch.float32)
    x_idx = torch.arange(t_x_max, device=device).unsqueeze(0)

    for y in range(1, t_y_max):
        x_lo = (t_x_lens + y - t_y_lens).clamp(min=0)
        x_hi = torch.minimum(t_x_lens, torch.tensor(y + 1, device=device))
        valid = (x_idx >= x_lo.unsqueeze(1)) & (x_idx < x_hi.unsqueeze(1))

        prev_row = value[:, y - 1, :]

        is_diag = x_idx == y
        v_cur = torch.where(is_diag, max_neg, prev_row)

        v_prev = torch.full_like(prev_row, -1e9)
        v_prev[:, 1:] = prev_row[:, :-1]

        best = torch.maximum(v_prev, v_cur)
        value[:, y, :] += torch.where(valid, best, torch.zeros_like(best))

    return value


def _mas_backward_eager(value, t_y_lens, t_x_lens, out_dtype):
    """Backward trace pass — eager (non-JIT) fallback."""
    device = value.device
    b, t_y_max, t_x_max = value.shape

    path = torch.zeros(b, t_y_max, t_x_max, device=device, dtype=out_dtype)
    index = t_x_lens - 1

    for y in range(t_y_max - 1, -1, -1):
        active = y < t_y_lens

        path[:, y, :].scatter_(
            1,
            index.clamp(min=0).unsqueeze(1),
            active.unsqueeze(1).to(out_dtype),
        )

        if y == 0:
            break

        idx_clamped = index.clamp(min=0)
        idx_prev = (index - 1).clamp(min=0)

        val_at_idx = value[:, y - 1, :].gather(1, idx_clamped.unsqueeze(1)).squeeze(1)
        val_at_idx_m1 = value[:, y - 1, :].gather(1, idx_prev.unsqueeze(1)).squeeze(1)

        do_dec = active & (index != 0) & ((index == y) | (val_at_idx < val_at_idx_m1))
        index = index - do_dec.long()

    return path


def maximum_path(neg_cent, mask):
    """Monotonic Alignment Search — GPU-native (default) with Cython fallback.

    Uses a pure-PyTorch implementation (JIT-compiled when possible) that runs
    entirely on the device of ``neg_cent`` (typically GPU), eliminating the
    CPU-GPU data transfers that bottleneck the Cython path.

    Args:
        neg_cent: [batch, t_y, t_x] tensor
        mask: [batch, t_y, t_x] tensor

    Returns:
        Path tensor [batch, t_y, t_x]
    """
    return maximum_path_pytorch(neg_cent, mask)


# ------------------------------------------------------------------ #
#  Cython fallback (renamed, kept for compatibility / testing)
# ------------------------------------------------------------------ #


def _maximum_path_cython(neg_cent, mask):
    """Cython optimized version with adaptive chunked processing.

    This is the original implementation retained as a fallback.  It
    transfers data to CPU, runs the Cython kernel, and copies back.

    Args:
        neg_cent: [batch, t_t, t_s] tensor
        mask: [batch, t_t, t_s] tensor

    Returns:
        Path tensor [batch, t_t, t_s]
    """
    if maximum_path_c is None:
        raise ImportError(
            "Cython extension 'monotonic_align.core' is not built. "
            "Run: python piper_train/vits/monotonic_align/setup.py build_ext --inplace"
        )
    device = neg_cent.device
    dtype = neg_cent.dtype
    batch_size, t_t, t_s = neg_cent.shape
    matrix_size = t_t * t_s

    # 大きな行列の場合は1つずつ処理
    if matrix_size > _LARGE_MATRIX_THRESHOLD:
        max_chunk = 1
    else:
        max_chunk = 10

    # バッチをチャンクに分割
    if batch_size > max_chunk:
        results = []
        for i in range(0, batch_size, max_chunk):
            chunk_end = min(i + max_chunk, batch_size)
            chunk_result = _maximum_path_core(
                neg_cent[i:chunk_end], mask[i:chunk_end], device, dtype
            )
            results.append(chunk_result)
        return torch.cat(results, dim=0)

    return _maximum_path_core(neg_cent, mask, device, dtype)


def _maximum_path_core(neg_cent, mask, device, dtype):
    """Core Cython implementation without chunking."""
    neg_cent_np = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent_np.shape, dtype=np.int32)
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
