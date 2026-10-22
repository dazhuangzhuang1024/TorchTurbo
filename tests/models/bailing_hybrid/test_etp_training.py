"""Real registered Bailing language and image FP32 engine acceptance."""

import json
from dataclasses import replace

import pytest
import torch
import torch.multiprocessing as mp

from .etp_helpers import (LANGUAGE_CASES, auxiliary, expert_function, literal_routing,
                          ordinary_model, run_bailing_case, write_bailing_source)
from .tp_helpers import cpu_bailing_leaves
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.experts import GroupedMLP


@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_routing_selected_sigmoid_group_matrix(family):
    case = replace(LANGUAGE_CASES[0], family=family)
    with cpu_bailing_leaves():
        gate = ordinary_model(case).model.layers[1].mlp.gate
        with torch.no_grad():
            gate.weight.copy_(torch.linspace(-.4, .7, 128).reshape(8, 16))
            gate.expert_bias.copy_(torch.linspace(-.035, .035, 8))
        hidden = torch.sin(torch.arange(80).reshape(1, 5, 16).float() * .13)
        # Global top2 spans groups; grouped routing selects two from group 1.
        with torch.no_grad():
            gate.weight[:, 0] = torch.tensor([3., -2., -2., -2., 1.7, 1.6, -2., -2.])
        hidden[0, 0].zero_()
        hidden[0, 0, 0] = 1.
        raw_ids, raw_probs, logits = gate(hidden)
        ids, probabilities = literal_routing(logits, gate.expert_bias)
        global_ids = (logits[0].float().sigmoid() + gate.expert_bias).topk(2).indices
        assert set(global_ids.tolist()) == {0, 4}
        assert set(ids[0].tolist()) == {4, 5}
        assert torch.equal(raw_ids, ids)
        torch.testing.assert_close(raw_probs, probabilities.gather(-1, ids), rtol=0, atol=0)
        assert gate.top_k == 2 and gate.topk_group == 1
        router_config = MoEConfig(hidden_size=16, num_moe_experts=8, perform_initialization=False,
            moe_router_num_groups=2, moe_router_group_topk=1, moe_router_topk=2,
            moe_router_topk_scaling_factor=1.3, moe_router_score_function="sigmoid",
            moe_router_dtype="fp32", moe_router_enable_expert_bias=True)
        prepared = TopKRouter(router_config)
        prepared.weight.data.copy_(gate.weight)
        prepared.expert_bias.copy_(gate.expert_bias)
        actual_probs, actual_map, actual_logits = prepared(hidden)
        assert torch.equal(actual_map, torch.nn.functional.one_hot(ids, 8).sum(1).bool())
        torch.testing.assert_close(actual_logits.reshape_as(logits), logits, rtol=0, atol=0)
        torch.testing.assert_close(actual_probs, probabilities, rtol=0, atol=0)


def test_routing_clamp_after_silu_and_valid_token_auxiliary():
    with cpu_bailing_leaves():
        model = ordinary_model(LANGUAGE_CASES[0])
        for expert, limit in ((model.model.layers[1].mlp.experts[0], .4),
                              (model.model.layers[1].mlp.shared_experts, .5)):
            with torch.no_grad():
                expert.gate_proj.weight.fill_(.5)
                expert.up_proj.weight.fill_(-.5)
                expert.down_proj.weight.fill_(.25)
            hidden = torch.ones(1, 2, 16)
            gate = torch.nn.functional.silu(torch.tensor(8.))
            assert gate > limit
            expected = torch.full((1, 2, 16), -limit ** 2 * 12 * .25)
            torch.testing.assert_close(expert_function(expert, hidden, limit), expected)
            if limit == .5:
                torch.testing.assert_close(expert(hidden), expected)
            else:
                grouped_config = MoEConfig(hidden_size=16, num_moe_experts=1,
                    moe_ffn_hidden_size=12, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                    perform_initialization=False, moe_swiglu_limit=[limit],
                    moe_swiglu_clamp_gate_after_activation=True)
                grouped = GroupedMLP(1, False, grouped_config, layer_number=0)
                with torch.no_grad():
                    grouped.weight1.copy_(torch.cat((expert.gate_proj.weight.t(), expert.up_proj.weight.t()), 1))
                    grouped.weight2.copy_(expert.down_proj.weight.t())
                actual, _ = grouped(hidden.reshape(-1, 16), torch.tensor([2]), torch.ones(2))
                torch.testing.assert_close(actual.reshape_as(expected), expected)
            wrong = torch.nn.functional.silu(torch.tensor(limit)) * -limit * 12 * .25
            assert abs(float(expected[0, 0, 0]) - float(wrong)) > .1
        zero_aux, zero_z, counts = auxiliary(torch.empty(0, 8))
        assert zero_aux == zero_z == 0 and counts.sum() == 0


@pytest.mark.parametrize("case", LANGUAGE_CASES,
                         ids=["hybrid-data-ep-aux-z", "vl-sp-chunk-aux-z", "hybrid-sp-ep"])
def test_registered_language_training(tmp_path, case):
    write_bailing_source(tmp_path / "source", case)
    mp.spawn(run_bailing_case, args=(str(tmp_path), case), nprocs=case.world, join=True)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(case.world)]
    assert all(record["complete"] for record in records)
    if case.ep > 1:
        assert any(sum(route["counts"][:4]) == 0 or sum(route["counts"][4:]) == 0
                   for record in records for route in record["routes"])
        for owner in range(case.ep):
            assert any(sum(route["counts"][owner*4:(owner+1)*4]) > 0
                       for record in records for route in record["routes"])


def test_registered_vl_image_training(tmp_path):
    case = replace(LANGUAGE_CASES[1], image=True)
    write_bailing_source(tmp_path / "source", case)
    mp.spawn(run_bailing_case, args=(str(tmp_path), case), nprocs=case.world, join=True)
    assert all(json.loads((tmp_path / f"rank-{rank}.json").read_text())["complete"] for rank in range(case.world))
