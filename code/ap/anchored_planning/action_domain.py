"""Fresh policy action projection, using the actual environment Box."""
import numpy as np


def counts(original, effective):
    changed = np.asarray(original) != np.asarray(effective)
    if changed.size == 0:  # Clean condition or a prefix stopped before its first action.
        return dict(chunks=0, action_steps=0, coordinates=0)
    return dict(chunks=int(changed.reshape(-1, *changed.shape[-2:]).any(axis=(1, 2)).sum()),
                action_steps=int(changed.any(axis=-1).sum()), coordinates=int(changed.sum()))


def project(raw, low, high):
    """Preserve already legal FP32 values exactly; return effective raw actions."""
    original = np.asarray(raw, np.float32)
    effective = np.clip(original, low, high).astype(np.float32)
    return effective, counts(original, effective)


def prepare_candidates(raw, low, high, scaler):
    effective, report = project(raw, low, high)
    normalized = scaler.transform(effective.reshape(-1, effective.shape[-1])).astype(np.float32).reshape(effective.shape)
    return effective, normalized, report
