"""Keep structural failures outside bounded BF16 numerical qualification."""

import pytest
import torch

from .precision_helpers import (
    NumericalQualificationError, PrecisionEvidence, finish_numerical_qualification,
)


@pytest.mark.parametrize("name", ["stage 0.post_attention_layernorm",
    "stage 1.input_layernorm", "stage 1.post_attention_layernorm", "stage final"])
def test_known_finite_stage_deviation_is_collected(name):
    evidence = PrecisionEvidence()
    evidence.close(torch.ones(2) * 1.2, torch.ones(2), name)
    assert evidence.failures
    with pytest.raises(NumericalQualificationError):
        finish_numerical_qualification(evidence.failures)


@pytest.mark.parametrize("name", ["stage 0.post_attention_layernorm",
    "stage 1.input_layernorm", "stage 1.post_attention_layernorm", "stage final"])
def test_known_stage_shape_mismatch_is_an_ordinary_assertion(name):
    evidence = PrecisionEvidence()
    with pytest.raises(AssertionError) as caught:
        evidence.close(torch.ones(2), torch.ones(1), name)
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["stage 0.post_attention_layernorm",
    "stage 1.input_layernorm", "stage 1.post_attention_layernorm", "stage final"])
@pytest.mark.parametrize("operand", ["actual", "expected"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_known_stage_nonfinite_operand_is_an_ordinary_assertion(name, operand, invalid):
    evidence = PrecisionEvidence()
    left, right = torch.ones(1), torch.ones(1)
    (left if operand == "actual" else right)[0] = invalid
    with pytest.raises(AssertionError) as caught:
        evidence.close(left, right, name)
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["gradient x", "stage x", "optimizer x exp_avg"])
def test_shape_mismatch_is_an_ordinary_assertion(name):
    evidence = PrecisionEvidence()
    with pytest.raises(AssertionError) as caught:
        evidence.close(torch.ones(2), torch.ones(1), name)
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["parameter x", "K scaled input 0", "stage x"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("operand", ["actual", "expected"])
def test_nonfinite_operand_is_an_ordinary_assertion(name, invalid, operand):
    evidence = PrecisionEvidence()
    left, right = torch.ones(1), torch.ones(1)
    (left if operand == "actual" else right)[0] = invalid
    with pytest.raises(AssertionError) as caught:
        evidence.close(left, right, name)
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


def test_optimizer_step_mismatch_is_an_ordinary_assertion():
    evidence = PrecisionEvidence()
    with pytest.raises(AssertionError) as caught:
        evidence.close(torch.tensor(2.), torch.tensor(1.), "optimizer x step")
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["stage x", "logits", "probabilities 0", "main report"])
def test_unexpected_finite_numerical_failure_is_an_ordinary_assertion(name):
    evidence = PrecisionEvidence()
    with pytest.raises(AssertionError) as caught:
        evidence.close(torch.ones(2) * 2, torch.ones(2), name)
    assert type(caught.value) is AssertionError
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["gradient x", "clipped x", "K scaled input 0",
    "update x", "optimizer x exp_avg", "parameter x", "norm"])
def test_declared_finite_tolerance_failure_is_collected(name):
    evidence = PrecisionEvidence()
    evidence.close(torch.ones(2) * 1.2, torch.ones(2), name)
    assert evidence.failures
    assert all(failure[0] == name for failure in evidence.failures)


@pytest.mark.parametrize("shape", [(2,), (0,)])
@pytest.mark.parametrize("name", ["gradient x", "update x", "optimizer x exp_avg",
    "optimizer x exp_avg_sq"])
def test_exact_zero_or_empty_pair_agrees(shape, name):
    evidence = PrecisionEvidence()
    evidence.close(torch.zeros(shape), torch.zeros(shape), name)
    assert evidence.failures == []


@pytest.mark.parametrize("name", ["gradient x", "update x", "optimizer x exp_avg"])
def test_nonzero_actual_with_zero_reference_is_collected(name):
    evidence = PrecisionEvidence()
    evidence.close(torch.ones(2), torch.zeros(2), name)
    assert evidence.failures


def test_typed_xfail_cannot_accept_an_ordinary_assertion():
    assert not isinstance(AssertionError("shape"), NumericalQualificationError)


@pytest.mark.parametrize("name", ["stage x", "logits", "probabilities 0", "optimizer x step"])
def test_parent_rejects_unexpected_failure_categories(name):
    with pytest.raises(AssertionError) as caught:
        finish_numerical_qualification([(name, "elementwise", 1.)])
    assert type(caught.value) is AssertionError


def test_parent_raises_typed_failure_only_for_declared_categories():
    finish_numerical_qualification([])
    with pytest.raises(NumericalQualificationError):
        finish_numerical_qualification([("gradient x", "vector", .2)])
