import torch
import torch.nn.functional as F

from .ttt_ops import silu_backprop, zeropower_via_newtonschulz5


def swiglu_l2_grads(w0, w1, w2, ki, vi, lr0i, lr1i, lr2i):
    """Descent directions (to be added) for one chunk under the l2 regression bias."""
    # w0, w2: [b, dh, dk]; w1: [b, dv, dh]; ki: [b, l, dk]; vi: [b, dv, l]; lr*: [b, l, 1|d]
    kT = ki.transpose(1, 2)                                  # [b, dk, l]

    gate_before_act = torch.bmm(w0, kT)                      # [b, dh, l]
    hidden_before_mul = torch.bmm(w2, kT)                    # [b, dh, l]
    gate = F.silu(gate_before_act, inplace=False)
    hidden = gate * hidden_before_mul                        # [b, dh, l]

    # err = v - pred must be computed in fp32; bf16 loses the near-equal difference.
    acc = torch.promote_types(torch.float32,
                              torch.promote_types(w1.dtype, vi.dtype))
    with torch.autocast(device_type=vi.device.type, enabled=False):
        pred = torch.bmm(w1.to(acc), hidden.to(acc))         # [b, dv, l]
        err = (vi.to(acc) - pred).to(vi.dtype)               # [b, dv, l]

    # upstream's backward pass, seeded with the residual instead of v
    dhidden = torch.bmm(w1.transpose(1, 2), err)
    dhidden_before_mul = dhidden * gate
    dgate_before_act = silu_backprop(dhidden * hidden_before_mul, gate_before_act)

    dw1 = torch.bmm(err, (hidden.transpose(1, 2) * lr1i).type_as(err))
    dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))
    return dw0, dw1, dw2


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu_l2(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,    # [b, s, 1]
    retention: torch.Tensor = None,   # [b, s, 1], alpha in (0, 1)
    return_state: bool = False,
    init_momentum: tuple = None,
    return_momentum: bool = False,
    return_trajectory: bool = False,
    return_decay: bool = False,
):
    """Drop-in replacement for `block_causal_lact_swiglu` with the l2 bias and retention gate."""
    # Readers must use the trajectory (state before each chunk); the final state leaks the future.
    if chunk_size < 1 or k.shape[1] < 1:
        raise ValueError('chunk_size and sequence length must be positive')
    if momentum is not None:
        # Incremental callers must pass init_momentum back in to match a single pass.
        if init_momentum is not None:
            dw0_momentum, dw1_momentum, dw2_momentum = init_momentum
        else:
            dw0_momentum = torch.zeros_like(w0)
            dw1_momentum = torch.zeros_like(w1)
            dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)   # [b, dk, l]
    v = v.transpose(1, 2)   # [b, dv, l]

    output = torch.zeros_like(v)
    traj0, traj1, traj2 = [], [], []
    decays = []

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        # apply first: weights fit on chunks strictly before this one
        if return_trajectory:
            traj0.append(w0)
            traj1.append(w1)
            traj2.append(w2)
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        dw0, dw1, dw2 = swiglu_l2_grads(w0, w1, w2, ki, vi, lr0i, lr1i, lr2i)

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            # Atlas Eq. 32: NS discards magnitude, so reapply lr outside it (chunk mean).
            eta0 = lr0i.mean(dim=1, keepdim=True)
            eta1 = lr1i.mean(dim=1, keepdim=True)
            eta2 = lr2i.mean(dim=1, keepdim=True)
            dw0 = zeropower_via_newtonschulz5(dw0).type_as(dw0) * eta0
            dw1 = zeropower_via_newtonschulz5(dw1).type_as(dw1) * eta1
            dw2 = zeropower_via_newtonschulz5(dw2).type_as(dw2) * eta2

        # Atlas Eq. 32: multiplicative decay in place of Eq. 8's projection.
        if retention is not None:
            a_i = retention[:, s_index:e_index, :].mean(dim=1, keepdim=True)
            if return_decay:
                decays.append(a_i)
            w0 = w0 * a_i + dw0
            w1 = w1 * a_i + dw1
            w2 = w2 * a_i + dw2
        else:
            if return_decay:
                decays.append(w0.new_ones((w0.shape[0], 1, 1)))
            w0 = w0 + dw0
            w1 = w1 + dw1
            w2 = w2 + dw2

    # tail chunk: read out with the final weights, no further update
    if return_trajectory:
        traj0.append(w0)
        traj1.append(w1)
        traj2.append(w2)
    s_index = e_index
    e_index = seq_len
    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    out = output.transpose(1, 2)
    if return_momentum:
        mom = ((dw0_momentum, dw1_momentum, dw2_momentum)
               if momentum is not None else None)
        ret = (out, w0, w1, w2, mom)
    elif return_state or return_trajectory or return_decay:
        ret = (out, w0, w1, w2)
    else:
        return out
    if return_trajectory:
        ret = ret + ((torch.stack(traj0), torch.stack(traj1),
                      torch.stack(traj2)),)
    if return_decay:
        ret = ret + (torch.stack(decays) if decays else w0.new_empty((0, w0.shape[0], 1, 1)),)
    return ret


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def read_lact_swiglu_l2(traj, q, chunk_size: int):
    """Read a shared memory trajectory with this layer's own queries."""
    T0, T1, T2 = traj
    n_blocks = T0.shape[0]
    seq_len = q.shape[1]
    if chunk_size < 1 or seq_len < 1:
        raise ValueError('chunk_size and sequence length must be positive')
    expected = (seq_len + chunk_size - 1) // chunk_size
    if any(t.shape[0] != expected for t in traj):
        raise ValueError(
            f'trajectory must have {expected} blocks for {seq_len} positions at chunk '
            f'{chunk_size}, got {[t.shape[0] for t in traj]}. The writer and '
            'the reader must run on the same sequence length.'
        )
    q = q.transpose(1, 2)                                    # [b, dk, l]
    blocks = []
    for c in range(n_blocks):
        s_index = c * chunk_size
        e_index = min(s_index + chunk_size, seq_len)
        qi = q[:, :, s_index:e_index]
        gate = F.silu(torch.bmm(T0[c], qi), inplace=True)
        blocks.append(torch.bmm(T1[c], gate * torch.bmm(T2[c], qi)))
    return torch.cat(blocks, dim=2).transpose(1, 2)


__all__ = ['block_causal_lact_swiglu_l2', 'read_lact_swiglu_l2',
           'swiglu_l2_grads']
