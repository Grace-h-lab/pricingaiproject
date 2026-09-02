"""Small PyTorch device helpers used by the experiment runners.

Device policy
-------------
The default is *auto*: use CUDA when a GPU is visible, otherwise CPU.
Override explicitly with the ``PRICING_DT_DEVICE`` environment variable, e.g.

    PRICING_DT_DEVICE=cpu    python run.py ...    # force reproducible CPU
    PRICING_DT_DEVICE=cuda   python run.py ...    # force GPU (errors if absent)
    PRICING_DT_DEVICE=cuda:1 python run.py ...    # pick a specific GPU

Numerics note: the experiments are seeded per run and every reported quantity is
an *exact* expected value from the simulator's dynamic-programming tables, not a
Monte-Carlo estimate. Float32 matmul ordering still differs between CPU and CUDA,
so trained network weights are not bit-identical across devices; TF32 is disabled
below so the GPU keeps full float32 mantissa precision and the device-to-device
drift stays at the level of ordinary float32 round-off.
"""
import os

import torch

_ENV_VAR = "PRICING_DT_DEVICE"
_configured = False


def _configure_backend():
    """Keep CUDA matmuls at true float32 (no TF32) so GPU runs stay comparable
    to the CPU reference numbers rather than silently dropping to 10-bit
    mantissas on Ampere-and-later cards."""
    global _configured
    if _configured or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except AttributeError:      # very old torch
        pass
    _configured = True


def default_device():
    """Auto-select CUDA when available; honour an explicit env override."""
    override = os.environ.get(_ENV_VAR)
    if override:
        dev = torch.device(override)
        if dev.type == "cuda":
            _configure_backend()
        return dev
    if torch.cuda.is_available():
        _configure_backend()
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device=None, model=None):
    """Resolve an explicit device, or infer one from a model, or auto-select."""
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cuda":
            _configure_backend()
        return dev
    if model is not None:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            pass
    return default_device()


def device_report():
    """One-line human-readable description of the device actually in use."""
    dev = default_device()
    if dev.type == "cuda":
        idx = dev.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        src = "env override" if os.environ.get(_ENV_VAR) else "auto-detected"
        return f"device={dev} ({name}, {total:.1f} GiB, {src}, tf32=off)"
    src = "env override" if os.environ.get(_ENV_VAR) else "no CUDA visible"
    return f"device={dev} ({src})"


def align_inputs(module, *tensors):
    """Move incoming tensors onto ``module``'s parameter device.

    The diagnostics routinely build small probe tensors with plain
    ``torch.tensor(...)`` calls and feed them straight to a trained model. Now
    that models stay resident on the GPU instead of being copied back to the CPU
    after training, every such site would otherwise raise a device-mismatch
    error. Having the models align their own inputs keeps those call sites
    working untouched. A tensor already on the right device is returned as-is,
    so CPU-only runs are completely unaffected.
    """
    try:
        dev = next(module.parameters()).device
    except (AttributeError, StopIteration):
        return tensors[0] if len(tensors) == 1 else tensors
    out = tuple(t.to(dev) if isinstance(t, torch.Tensor) and t.device != dev else t
                for t in tensors)
    return out[0] if len(out) == 1 else out


def dataloader_permutation(n):
    """Reproduce one epoch of ``DataLoader(..., shuffle=True)`` ordering exactly.

    The training loops index a device-resident tensor rather than iterating a
    DataLoader, whose per-sample Python collation and per-batch host-to-device
    copy dominate the step time for the small datasets in this study. The
    minibatch composition must nevertheless match a DataLoader's exactly, or
    every trained model, and so every reported number, would shift.

    Two global-RNG draws happen per epoch, in this order:

      1. ``_BaseDataLoaderIter.__init__`` pulls an int64 ``_base_seed`` (only used
         to seed worker processes, so irrelevant here — but it still advances the
         global stream, so it has to be replayed);
      2. ``RandomSampler.__iter__`` (with ``generator=None``) pulls a second int64
         seed, seeds a *fresh* generator with it, and permutes with that — the
         permutation therefore does NOT come from the global RNG directly.

    Replaying both keeps the global RNG stream and the batch order identical, which
    was verified to reproduce a real DataLoader's index sequence exactly.
    """
    torch.empty((), dtype=torch.int64).random_()          # DataLoader's _base_seed
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randperm(n, generator=gen)
