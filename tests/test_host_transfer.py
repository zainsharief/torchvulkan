import pytest
import torch
import torchvulkan  # noqa: F401

DEVICE = "vulkan"
MB = 1 << 20


@pytest.mark.parametrize("nbytes", [1 * MB, 4 * MB, 4 * MB + 4, 64 * MB + 12345 * 4, 300 * MB])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_round_trip(nbytes, dtype):
    n = nbytes // torch.empty((), dtype=dtype).element_size()
    g = torch.Generator().manual_seed(n)
    x = torch.randint(-100, 100, (n,), generator=g).to(dtype)
    v = x.to(DEVICE)
    assert torch.equal(v.cpu(), x)
    # a second copy into an existing tensor
    y = torch.empty_like(x)
    y.copy_(v)
    assert torch.equal(y, x)


@pytest.mark.parametrize("offset", [1, 7, 1023, 4096 + 3])
def test_unaligned_host_slices(offset):
    # host pointers that don't start on a page: views into a larger tensor, in both directions
    base = torch.arange(8 * MB + 2 * offset, dtype=torch.int32)
    src = base[offset:offset + 8 * MB]
    v = src.to(DEVICE)
    assert torch.equal(v.cpu(), src)

    out = torch.full((8 * MB + 2 * offset,), -1, dtype=torch.int32)
    out[offset:offset + 8 * MB].copy_(v)
    assert torch.equal(out[offset:offset + 8 * MB], src)
    # the bytes around the destination range stay untouched
    assert (out[:offset] == -1).all() and (out[offset + 8 * MB:] == -1).all()


def test_source_can_change_after_upload():
    x = torch.ones(16 * MB, dtype=torch.float32)
    v = x.to(DEVICE)
    x.fill_(2.0)
    assert (v.cpu() == 1.0).all()


def test_device_values_written_before_download_are_seen():
    v = torch.zeros(16 * MB, device=DEVICE)
    v.add_(3.0)  # pending GPU work the download has to wait for
    assert (v.cpu() == 3.0).all()


def test_read_only_file_mapping(tmp_path):
    # torch.from_file maps the file; the import may be refused, the staging path then takes over
    path = tmp_path / "data.bin"
    data = torch.arange(6 * MB, dtype=torch.int32)
    data.numpy().tofile(path)
    mapped = torch.from_file(str(path), shared=False, size=6 * MB, dtype=torch.int32)
    assert torch.equal(mapped.to(DEVICE).cpu(), data)
