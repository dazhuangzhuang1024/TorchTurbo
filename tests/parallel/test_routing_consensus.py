"""Integer ETP routing consensus preserves each lane's probability autograd."""

from datetime import timedelta
import inspect

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity
from torchturbo.models.modules.moe.moe_utils import compute_routing_scores_for_aux_loss


@pytest.mark.parametrize('score_function', ['sigmoid', 'sqrtsoftplus'])
def test_legacy_grouped_bias_uses_unbiased_probability_gradients(score_function):
    base = torch.tensor([
        [1.5, -2.1, -2.2, -2.3], [1.4, 1.3, -2.4, -2.5],
        [1.2, 1.1, -2.6, -2.7], [1., .9, -2.8, -2.9],
        [.8, .7, .6, .5], [.4, .3, .2, .1],
        [0., -.1, -.2, -.3], [-.4, -.5, -.6, -.7],
    ]).flatten()
    logits = torch.stack([base, base.roll(4), base.roll(8)]).requires_grad_()
    bias = torch.tensor([0.] * 20 + [.25] * 4 + [.35] * 4 + [.45] * 4).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    if score_function == 'sigmoid':
        scores = reference.sigmoid()
    else:
        scores = torch.nn.functional.softplus(reference).sqrt()

    def select_groups_and_experts(values):
        group_masks, expert_masks = [], []
        for row in values.detach().tolist():
            groups = [sum(sorted(row[start:start + 4], reverse=True)[:2])
                      for start in range(0, 32, 4)]
            chosen_groups = sorted(range(8), key=lambda group: groups[group], reverse=True)[:4]
            candidates = [expert for expert in range(32) if expert // 4 in chosen_groups]
            chosen_experts = sorted(candidates, key=lambda expert: row[expert], reverse=True)[:8]
            group_masks.append([group in chosen_groups for group in range(8)])
            expert_masks.append([expert in chosen_experts for expert in range(32)])
        return torch.tensor(group_masks), torch.tensor(expert_masks)

    expected_groups, expected_map = select_groups_and_experts(scores + bias.detach())
    unbiased_groups, unbiased_map = select_groups_and_experts(scores)
    assert not torch.equal(expected_groups, unbiased_groups)
    assert not torch.equal(expected_map, unbiased_map)
    ungrouped_map = torch.zeros_like(expected_map).scatter(
        1, (scores.detach() + bias.detach()).topk(8, dim=-1).indices, True)
    assert not torch.equal(expected_map, ungrouped_map)
    selected = scores.masked_select(expected_map).view(3, 8)
    expected = torch.zeros_like(reference).masked_scatter(
        expected_map, selected / selected.sum(-1, keepdim=True) * 2.5)
    probabilities, routing_map, counts = topk_softmax_with_capacity(
        logits, topk=8, num_groups=8, group_topk=4,
        score_function=score_function, expert_bias=bias,
        scaling_factor=2.5, routing_group=None,
    )
    torch.testing.assert_close(routing_map, expected_map)
    torch.testing.assert_close(counts, expected_map.sum(0))
    assert torch.all(routing_map.sum(-1) == 8)
    assert not torch.any(routing_map.view(3, 8, 4).any(-1) & ~expected_groups)
    torch.testing.assert_close(probabilities, expected)
    factors = torch.linspace(.2, 1.7, 96).view(3, 32)
    (probabilities * factors).sum().backward()
    (expected * factors).sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert torch.count_nonzero(logits.grad[~expected_map]) == 0
    assert bias.grad is None


def _routing_worker(rank, path):
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        assert 'routing_group' in inspect.signature(topk_softmax_with_capacity).parameters
        from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
        context = ParallelContext('cpu', sp_size=2, ep_size=2, tp_size=2, etp_size=2)
        lane = rank % 2
        group = context.etp_mesh.get_group()
        for score_function, pre in [('softmax', False), ('softmax', True), ('sigmoid', False), ('sqrtsoftplus', False)]:
            for topk in (1, 2):
                for selection in ('plain', 'bias', 'group', 'hash', 'empty'):
                    num_rows = 0 if selection == 'empty' else 3
                    base = torch.tensor([[.01, .02, .03, .04], [.04, .03, .02, .01], [.01, .03, .04, .02]])[:num_rows]
                    base = base.roll(rank//2, dims=1)
                    logits = (base + lane * torch.tensor([.1, -.1, -.1, .1])).requires_grad_()
                    args = dict(topk=topk, use_pre_softmax=pre, score_function=score_function, scaling_factor=1.7)
                    if selection == 'bias' and score_function != 'softmax':
                        args['expert_bias'] = torch.tensor([-.2, -.3, .4, .5])
                    if selection == 'group':
                        args.update(num_groups=2, group_topk=1)
                    if selection == 'hash':
                        args.update(use_hash_routing=True, hash_table=torch.tensor([[0, 1], [1, 2], [2, 3]])[:, :topk], input_ids=torch.arange(num_rows))
                    _, canonical, _ = topk_softmax_with_capacity(base, **args)
                    probs, routing, counts = topk_softmax_with_capacity(logits, **args, routing_group=group)
                    torch.testing.assert_close(routing, canonical)
                    torch.testing.assert_close(counts, canonical.sum(0))
                    reference = logits.detach().clone().requires_grad_()
                    if score_function == 'softmax':
                        scores = reference.softmax(-1) if pre or selection == 'hash' else reference
                    elif score_function == 'sigmoid':
                        scores = reference.sigmoid()
                    else:
                        scores = torch.nn.functional.softplus(reference).sqrt()
                    selected = scores[canonical].view(num_rows, topk)
                    if score_function == 'softmax' and not pre and selection != 'hash':
                        selected = selected.softmax(-1)
                    elif score_function != 'softmax' and topk > 1:
                        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
                    expected = torch.zeros_like(reference).masked_scatter(canonical, selected * 1.7)
                    torch.testing.assert_close(probs, expected)
                    factors = torch.arange(4.).expand_as(probs)
                    (probs * factors).sum().backward()
                    (expected * factors).sum().backward()
                    torch.testing.assert_close(logits.grad, reference.grad)
                    padding = torch.tensor([False, True, False])[:num_rows]
                    aux_map, aux_scores = compute_routing_scores_for_aux_loss(logits, topk, score_function, padding, routing_group=group)
                    expected_aux, _ = compute_routing_scores_for_aux_loss(base, topk, score_function, padding)
                    torch.testing.assert_close(aux_map, expected_aux)
                    _, local_scores = compute_routing_scores_for_aux_loss(logits, topk, score_function, padding)
                    torch.testing.assert_close(aux_scores, local_scores)
        from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
        from torchturbo.models.modules.moe.moe_config import MoEConfig
        from torchturbo.models.modules.moe.router import TopKRouter
        config = MoEConfig(hidden_size=4, num_moe_experts=4, moe_router_topk=2,
            moe_router_score_function='sigmoid', moe_router_enable_expert_bias=True,
            moe_aux_loss_coeff=.2, moe_z_loss_coeff=.03)
        router = TopKRouter(config, sp_group=context.sp_mesh.get_group(), routing_group=group)
        router.expert_bias.copy_(torch.tensor([0., 0., .5, .5]))
        padding = torch.tensor([False, True, False])
        for use_ac in (False, True):
            for step in range(2):
                bases = [torch.tensor([[.04, .03, .02, .01], [.01, .02, .04, .03], [.03, .04, .02, .01]])
                         .roll(block + step, dims=1) for block in range(2)]
                base = bases[rank//2]
                logits = (base + lane * torch.tensor([-.02, .02, -.02, .02])).view(3, 1, 4).requires_grad_()
                with set_checkpoint_early_stop(False):
                    def route(value):
                        return router.routing(value, padding_mask=padding)[0]
                    probs = checkpoint(route, logits, use_reentrant=False) if use_ac else route(logits)
                ref_logits = logits.detach().clone().requires_grad_()
                scores = ref_logits.reshape(3, 4).sigmoid()
                dispatch_ids = (base.sigmoid() + router.expert_bias).topk(2, dim=-1).indices
                aux_ids = base.topk(2, dim=-1).indices
                assert not torch.equal(dispatch_ids.sort(-1).values, aux_ids.sort(-1).values)
                selected = scores.gather(1, dispatch_ids)
                expected = torch.zeros_like(scores).scatter(1, dispatch_ids, selected/selected.sum(-1, keepdim=True))
                torch.testing.assert_close(probs, expected)
                aux_scores = scores/scores.sum(-1, keepdim=True)
                aux_scores = aux_scores * (~padding)[:, None]
                global_counts = torch.zeros(4)
                for other in bases:
                    other_map = torch.zeros_like(other).scatter(1, other.topk(2, dim=-1).indices, 1)
                    global_counts += (other_map * (~padding)[:, None]).sum(0)
                aux = .2 * (aux_scores.sum(0) * global_counts).sum() * 4 * 2 / (2 * 4 * 4)
                z_loss = ref_logits.reshape(3, 4).logsumexp(-1).square()[~padding].mean() * .03
                factors = torch.arange(4.).expand_as(probs)
                (probs * factors).sum().backward()
                ((expected * factors).sum() + aux + z_loss).backward()
                torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-5, atol=2e-6)
    finally:
        dist.destroy_process_group()


def test_local_probability_gradients_with_integer_consensus(tmp_path):
    mp.spawn(_routing_worker, args=(str(tmp_path / 'store'),), nprocs=4, join=True)
