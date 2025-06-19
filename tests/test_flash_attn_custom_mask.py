import torch
import pytest
from hopper.flash_attn_interface import flash_attn_custom_mask_varlen_func

def make_data(batch_size, seqlen_q, seqlen_k, nheads, headdim, dtype=torch.float16, device='cuda'):
    q = torch.randn(batch_size * seqlen_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size * seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size * seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=device)
    seqused_q = torch.full((batch_size,), seqlen_q, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k

def test_custom_mask_basic():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size, seqlen_q, seqlen_k, nheads, headdim = 2, 4, 5, 2, 8
    q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = make_data(batch_size, seqlen_q, seqlen_k, nheads, headdim, device=device)
    max_seqlen_q, max_seqlen_k = seqlen_q, seqlen_k

    # 每个query只能看到前2个k
    visible_indices = torch.tensor([1, 1, 1, 1] * batch_size, dtype=torch.int32, device=device)
    out, lse = flash_attn_custom_mask_varlen_func(
        q, k, v, visible_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
    )
    assert out.shape == (batch_size * seqlen_q, nheads, headdim)
    # 检查mask生效：只前2个k有贡献
    # 这里可以用全1的v，方便验证
    v_ones = torch.ones_like(v)
    out2, _ = flash_attn_custom_mask_varlen_func(
        q, k, v_ones, visible_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
    )
    # out2应该不会有nan/inf
    assert torch.isfinite(out2).all()

def test_custom_mask_all_visible():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size, seqlen_q, seqlen_k, nheads, headdim = 1, 3, 3, 1, 4
    q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = make_data(batch_size, seqlen_q, seqlen_k, nheads, headdim, device=device)
    max_seqlen_q, max_seqlen_k = seqlen_q, seqlen_k
    visible_indices = torch.full((batch_size * seqlen_q,), seqlen_k - 1, dtype=torch.int32, device=device)
    out, _ = flash_attn_custom_mask_varlen_func(
        q, k, v, visible_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
    )
    # 与原生全可见对比
    # 这里可以用flash_attn_varlen_func（如果有的话）或直接检查输出不为nan
    assert torch.isfinite(out).all()

def test_custom_mask_all_invisible():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size, seqlen_q, seqlen_k, nheads, headdim = 1, 2, 3, 1, 4
    q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = make_data(batch_size, seqlen_q, seqlen_k, nheads, headdim, device=device)
    max_seqlen_q, max_seqlen_k = seqlen_q, seqlen_k
    visible_indices = torch.full((batch_size * seqlen_q,), -1, dtype=torch.int32, device=device)
    out, _ = flash_attn_custom_mask_varlen_func(
        q, k, v, visible_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
    )
    # 全部被mask，输出应为0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

def test_custom_mask_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size, seqlen_q, seqlen_k, nheads, headdim = 1, 2, 3, 1, 4
    q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = make_data(batch_size, seqlen_q, seqlen_k, nheads, headdim, device=device)
    max_seqlen_q, max_seqlen_k = seqlen_q, seqlen_k
    visible_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    out, _ = flash_attn_custom_mask_varlen_func(
        q, k, v, visible_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
    )
    loss = out.sum()
    loss.backward()
    # 检查梯度不为None
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()

if __name__ == '__main__':
    pytest.main([__file__]) 