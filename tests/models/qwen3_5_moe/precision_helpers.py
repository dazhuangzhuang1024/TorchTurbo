"""Predeclared bounded BF16 gates and non-vacuous logical block evidence."""

import torch

from tests.models.qwen3_moe.etp_helpers import Evidence, full


class CompleteEvidence(Evidence):
    def __init__(self):
        super().__init__()
        self.magnitudes = {}
        self.relative_errors = {}
        self.expert_magnitudes = {}
        self.target_counts = []

    def close(self, actual, expected, name, **kwargs):
        left, right = full(actual).double(), full(expected).double()
        magnitude = float(right.norm())
        self.magnitudes.setdefault(name, []).append(magnitude)
        relative = float((left - right).norm()) / magnitude if magnitude else 0.
        self.relative_errors[name] = max(self.relative_errors.get(name, 0.), relative)
        super().close(actual, expected, name, **kwargs)


class NumericalQualificationError(AssertionError):
    """Only completed dense BF16 numerical qualification failures."""


def numerical_failure_family(name):
    # Allow only these residual RMSNorm stages, where observed BF16 GDN rounding propagates.
    if name in ("stage 0.post_attention_layernorm", "stage 1.input_layernorm",
                "stage 1.post_attention_layernorm", "stage final"):
        return "known-stage"
    for family in ("gradient", "clipped", "K scaled", "update", "optimizer", "parameter"):
        if name.startswith(family + " ") and not name.endswith(" step"):
            return family
    return "norm" if name == "norm" else None


def finish_numerical_qualification(failures):
    unexpected = [failure for failure in failures if numerical_failure_family(failure[0]) is None]
    assert not unexpected, unexpected
    if failures:
        raise NumericalQualificationError(
            f"Completed BF16 dense qualification: {len(failures)} gate failures; rank JSON retains magnitudes/errors")


class PrecisionEvidence(CompleteEvidence):
    """BF16 compute/FP32 state gates, distinct from exact restore equality."""

    def __init__(self):
        super().__init__()
        self.failures = []

    def close(self, actual, expected, name, **kwargs):
        # BF16 ignores caller tolerances in favor of the predeclared gates.
        left, right = full(actual).double(), full(expected).double()
        assert left.shape == right.shape, (name, left.shape, right.shape)
        assert torch.isfinite(left).all(), (name, "actual is not finite")
        assert torch.isfinite(right).all(), (name, "expected is not finite")
        if name.endswith(" step"):
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
        self._close(left, right, name)
        if not name.startswith(("gradient ", "clipped ", "update ", "optimizer ")) or name.endswith(" step"):
            return
        if "experts.weight1" in name:
            left, right = left.reshape(-1, 16, 2, 12), right.reshape(-1, 16, 2, 12)
            for index in range(left.size(0)):
                self._close(left[index], right[index], f"{name}/expert{index}")
        elif "experts.weight2" in name:
            for index in range(left.size(0)):
                self._close(left[index], right[index], f"{name}/expert{index}")
        elif "self_attn.q_proj.weight" in name:
            for index, field in enumerate(("query", "gate")):
                self._close(left.reshape(4, 2, 8, 16)[:, index], right.reshape(4, 2, 8, 16)[:, index], f"{name}/{field}")
        elif "linear_attn.in_proj_qkv.weight" in name:
            for field, begin, end in (("query", 0, 16), ("key", 16, 32), ("value", 32, 64)):
                self._close(left[begin:end], right[begin:end], f"{name}/{field}")

    def _close(self, left, right, name):
        difference = left - right
        magnitude = float(right.norm())
        error = float(difference.abs().max()) if difference.numel() else 0.
        relative = float(difference.norm()) / magnitude if magnitude else 0.
        self.errors[name] = max(self.errors.get(name, 0.), error)
        self.magnitudes.setdefault(name, []).append(magnitude)
        self.relative_errors[name] = max(self.relative_errors.get(name, 0.), relative)
        if name.startswith(("gradient ", "clipped ", "K scaled")):
            bound = .05 if "probability" in name else .03
            if magnitude:
                if relative > bound:
                    self.failures.append((name, "vector", relative, bound, magnitude, error))
                scale = float((left * right).sum() / right.square().sum())
                self.scales.setdefault(name, []).append(scale)
                if not .95 < scale < 1.05:
                    self.failures.append((name, "scale", scale, magnitude, error))
            else:
                if error:
                    self.failures.append((name, "empty", error))
        elif name.startswith(("update ", "optimizer ")) and not name.endswith(" step"):
            if (magnitude and relative > .04) or (not magnitude and error):
                self.failures.append((name, "update/moment", relative, magnitude, error))
        else:
            if name.endswith(" step"):
                rtol, atol = 0., 0.
            elif name.startswith("parameter "):
                rtol, atol = .04, .0001
            elif name == "norm":
                rtol, atol = .03, .0007
            elif name == "main report":
                rtol, atol = .01, .03
            else:
                rtol, atol = .03, .03
            if numerical_failure_family(name) is None:
                torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=name)
            else:
                try:
                    torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=name)
                except AssertionError:
                    self.failures.append((name, "elementwise", relative, magnitude, error))

    def gradient(self, actual, expected, name):
        self.close(actual, expected, "gradient " + name)
