# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.attention.msa_m3_npu_new import minimax_m3_sparse_attn_decode


@patch("torch.ops._C_ascend.npu_sparse_attention_score", create=True)
def test_decode_uses_bfloat16_sparse_attention_score(mock_sparse_attention) -> None:
    q = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    key = torch.zeros(1, 128, 1, 4, dtype=torch.bfloat16)
    value = torch.ones(1, 128, 1, 4, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[[0, -1], [0, -1]]], dtype=torch.int32)
    select_num_idx = torch.tensor([[1, 1]], dtype=torch.int32)
    q_lens = torch.tensor([2], dtype=torch.int32)
    output = torch.empty_like(q)
    dequant_scale_buf = torch.ones(1, dtype=torch.float32)
    mock_sparse_attention.return_value = q + 1

    minimax_m3_sparse_attn_decode(
        q,
        (key, value),
        topk_idx,
        select_num_idx,
        torch.tensor([[0]], dtype=torch.int32),
        q_lens,
        torch.tensor([2], dtype=torch.int32),
        1,
        0.125,
        output,
        2,
        dequant_scale_buf,
        block_size=128,
    )

    args = mock_sparse_attention.call_args.args
    kwargs = mock_sparse_attention.call_args.kwargs
    assert args[0].dtype == torch.bfloat16
    assert args[1].dtype == torch.bfloat16
    assert args[2].dtype == torch.bfloat16
    assert kwargs["select_num_idx"] is select_num_idx
    assert kwargs["actual_seq_lengths"] is q_lens
    assert "q_dequant_scale" not in kwargs
    assert "k_dequant_scale" not in kwargs
    assert "v_dequant_scale" not in kwargs
    assert torch.equal(output, q + 1)
