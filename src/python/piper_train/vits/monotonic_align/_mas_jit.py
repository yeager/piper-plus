"""JIT-compiled Monotonic Alignment Search kernels.

Separated into its own module so ``torch.jit.script`` can access the
source code (required by TorchScript).
"""

import torch


@torch.jit.script
def mas_forward(
    value: torch.Tensor,
    t_y_lens: torch.Tensor,
    t_x_lens: torch.Tensor,
) -> torch.Tensor:
    """Forward DP pass -- vectorised across batch, sequential over rows.

    Modifies ``value`` **in-place** and returns it.
    """
    t_y_max = value.shape[1]
    t_x_max = value.shape[2]
    neg_inf = -1e9

    x_idx = torch.arange(t_x_max, device=value.device)  # [t_x_max]
    # Pre-allocate scratch buffers to avoid per-iteration allocation
    v_prev_buf = torch.empty_like(value[:, 0, :])  # [b, t_x_max]

    for y in range(1, t_y_max):
        x_lo = (t_x_lens + y - t_y_lens).clamp(min=0)  # [b]
        x_hi = torch.minimum(t_x_lens, torch.full_like(t_x_lens, y + 1))  # [b]
        valid = (x_idx >= x_lo.unsqueeze(1)) & (x_idx < x_hi.unsqueeze(1))  # [b, t_x_max]

        prev_row = value[:, y - 1, :]  # [b, t_x_max]

        # v_cur: value[y-1, x] when x != y, else -inf
        v_cur = prev_row.clone()
        if y < t_x_max:
            v_cur[:, y] = neg_inf

        # v_prev: value[y-1, x-1] when x > 0, else -inf
        v_prev_buf.fill_(neg_inf)
        v_prev_buf[:, 1:] = prev_row[:, :-1]

        best = torch.maximum(v_prev_buf, v_cur)
        # Masked in-place add
        value[:, y, :] += best * valid

    return value


@torch.jit.script
def mas_backward(
    value: torch.Tensor,
    t_y_lens: torch.Tensor,
    t_x_lens: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Backward trace pass -- vectorised across batch, sequential over rows."""
    b = value.shape[0]
    t_y_max = value.shape[1]
    t_x_max = value.shape[2]

    path = torch.zeros(b, t_y_max, t_x_max, device=value.device, dtype=out_dtype)
    index = t_x_lens - 1

    for y in range(t_y_max - 1, -1, -1):
        active = t_y_lens > y

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

        do_dec = active & (index != 0) & (
            (index == y) | (val_at_idx < val_at_idx_m1)
        )
        index = index - do_dec.long()

    return path
