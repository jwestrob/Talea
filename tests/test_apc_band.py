import pytest
import torch

from talea.inference import _clear_apc_diagonals


@pytest.mark.parametrize("length", [1, 3, 16, 73, 251])
@pytest.mark.parametrize("separation", [0, 1, 2, 6, 73])
@pytest.mark.parametrize("sparse", [True, False])
def test_in_place_band_matches_full_mask_and_diagonal(length, separation, sparse):
    # Signed zeros and non-finite values catch accidental arithmetic masking.
    x = torch.arange(2 * length**2, dtype=torch.float32).reshape(2, length, length).half()
    x[:, 0, 0] = float("nan")
    x[:, -1, 0] = -0.0
    reference = x.clone()
    pos = torch.arange(length)
    if separation > 0:
        reference = reference.masked_fill((pos[:, None] - pos[None, :]).abs() < separation, 0)
    reference.diagonal(dim1=-2, dim2=-1).fill_(0)
    original_storage = x.data_ptr()
    result = _clear_apc_diagonals(x, separation, sparse)
    assert original_storage == result.data_ptr()
    assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
