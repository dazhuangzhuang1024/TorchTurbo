"""Shared TP adapter contracts against the actual HF Trainer base."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer, TrainingArguments

from torchturbo.adapters.llamafactory_v0.trainer import make_torchturbo_trainer
from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import TurboTrainerMixin
from torchturbo.engine.core import DPBlockDistributedSampler, EpochDrivenDataLoader


class Engine:
    def __init__(self, gbs=6, tp_size=2):
        self.config = SimpleNamespace(
            gbs=gbs,
            backend_config=SimpleNamespace(
                cpu_offload=SimpleNamespace(enabled=False),
            ),
        )
        self.tp_size = tp_size
        self.calls = []
        self.collected = []

    def get_tp_info(self):
        return {'tp_size': self.tp_size}

    def get_dp_info(self):
        return {'dp_size': 2}

    def get_sp_info(self):
        return {'sp_size': 3}

    def resolve_gradient_accumulation_steps(self, framework_ga):
        return framework_ga

    def prepare_loss_step(self, model, samples, device):
        for sample in samples:
            assert not {'compute_loss_func', 'loss_scale', 'logits_to_keep'} & sample.keys()
        self.calls.append('prepare')
        return [dict(s, prepared=True) for s in samples], sum(int((s['labels'][:, 1:] != -100).sum()) for s in samples)

    def compute_prepared_loss(self, model, inputs, **kwargs):
        assert inputs['prepared']
        self.calls.append(kwargs)
        loss = model.weight.sum() * 4 / kwargs['num_items_in_batch']
        return loss, SimpleNamespace(loss=loss, main_loss=loss.detach() / 4)

    def collect_metrics(self, model):
        self.collected.append(model)


def make_adapter_class(framework, engine, stage='sft', base=Trainer):
    if framework == 'lf':
        return make_torchturbo_trainer(base, engine=engine, stage=stage)
    engine_registry.set_engine(engine, stage=stage)
    return type('SwiftHFTrainer', (TurboTrainerMixin, base), {})


class NativeLossDefaultsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swift templates can reset this flag after HF construction.
        self.model_accepts_loss_kwargs = False


def make(framework, engine, tmp_path):
    cls = make_adapter_class(framework, engine, base=NativeLossDefaultsTrainer)
    model = torch.nn.Linear(1, 1, bias=False)
    model.is_diffusion_model = False
    trainer = cls(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            gradient_accumulation_steps=3,
        ),
    )
    trainer.accelerator.backward = lambda loss, **kwargs: loss.backward()
    trainer.current_gradient_accumulation_steps = 2
    return trainer


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_fetch_count_backward_and_detached_step_contribution(framework, tmp_path):
    engine = Engine()
    trainer = make(framework, engine, tmp_path)
    try:
        source = [{'input_ids': torch.tensor([[1, 2, 3]]), 'labels': torch.tensor([[-100, 2, 3]])}, {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([[-100, -100]])}]
        samples, count = trainer.get_batch_samples(iter(source), 3, torch.device('cpu'))
        assert len(samples) == 2 and count == 2
        assert trainer.model_accepts_loss_kwargs
        returns = [trainer.training_step(trainer.model, s, num_items_in_batch=count) for s in samples]
        torch.testing.assert_close(trainer.model.weight.grad, torch.full_like(trainer.model.weight, 4))
        torch.testing.assert_close(sum(returns), trainer.model.weight.detach().sum())
        assert engine.calls[0] == 'prepare'
        assert all(call['gradient_accumulation_steps'] == 2 for call in engine.calls[1:])
        assert engine.collected == [trainer.model] * 2
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_count_hook_replaces_batches_in_place_after_framework_cleanup(framework, tmp_path):
    engine = Engine()
    trainer = make(framework, engine, tmp_path)
    original = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'labels': torch.tensor([[-100, 2, 3]]),
        'compute_loss_func': object(),
        'loss_scale': 0.5,
        'logits_to_keep': 1,
    }
    samples = [original]
    try:
        count = trainer._get_num_items_in_batch(samples, torch.device('cpu'))
        assert count == 2
        assert samples[0] is not original
        assert samples[0]['prepared']
        assert not {'compute_loss_func', 'loss_scale', 'logits_to_keep'} & samples[0].keys()
        assert {'compute_loss_func', 'loss_scale', 'logits_to_keep'} <= original.keys()
        torch.testing.assert_close(samples[0]['labels'], original['labels'])
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
@pytest.mark.parametrize('bound_engine', [False, True])
def test_non_tp_count_hook_preserves_native_kwargs_and_hf_count(
    framework, bound_engine, tmp_path,
):
    engine = Engine(tp_size=1) if bound_engine else None
    trainer = make(framework, engine, tmp_path)
    original = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'labels': torch.tensor([[5, -100, 7]]),
        'compute_loss_func': object(),
        'loss_scale': 0.5,
        'logits_to_keep': 1,
    }
    samples = [original]
    try:
        assert not trainer._tp_sft_enabled
        assert not trainer.model_accepts_loss_kwargs
        assert trainer._get_num_items_in_batch(samples, torch.device('cpu')) is None
        trainer.model_accepts_loss_kwargs = True
        assert trainer._get_num_items_in_batch(samples, torch.device('cpu')) == 2
        assert samples[0] is original
        assert {'compute_loss_func', 'loss_scale', 'logits_to_keep'} <= samples[0].keys()
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_logical_batch_sample_and_native_token_increment(framework, tmp_path):
    engine = Engine()
    trainer = make(framework, engine, tmp_path)
    try:
        assert trainer.get_total_train_batch_size(trainer.args) == trainer._train_batch_size * 3 * 2
        sampler = DPBlockDistributedSampler(list(range(13)), num_replicas=12, rank=0, dp_size=2, dp_rank=0, rank_block=6, shuffle=False, drop_last=True)
        loader = EpochDrivenDataLoader(sampler.dataset, sampler=sampler)
        assert trainer.num_examples(loader) == 12
        trainer.args.include_num_input_tokens_seen = 'all'
        trainer.accelerator.gather = lambda value: value.repeat(6)
        trainer.state.num_input_tokens_seen = 100
        trainer._track_num_input_tokens({'input_ids': torch.tensor([[1, 2, 3]])})
        assert trainer.state.num_input_tokens_seen == 103
        trainer._track_num_input_tokens({'input_ids': torch.tensor([[1, 2]])})
        assert trainer.state.num_input_tokens_seen == 105
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_activation_rejects_missing_gbs_and_non_sft(framework, tmp_path):
    engine = Engine()
    trainer = make(framework, engine, tmp_path)
    try:
        engine.config.gbs = None
        with pytest.raises(ValueError, match='tt_gbs'):
            trainer._activate_tp_sft(engine, 'sft')
        engine.config.gbs = 6
        for stage in ('dpo', 'rl', 'diffusion'):
            with pytest.raises(ValueError, match='SFT/PT'):
                trainer._activate_tp_sft(engine, stage)
        trainer.model.is_diffusion_model = True
        with pytest.raises(ValueError, match='diffusion'):
            trainer._activate_tp_sft(engine, 'sft')
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
@pytest.mark.parametrize('mode, expected', [('all', 4), ('non_padding', 2), ('no', 0)])
def test_native_token_modes_preserve_resume_and_ignore_supervision(framework, tmp_path, mode, expected):
    trainer = make(framework, Engine(), tmp_path)
    try:
        trainer.args.include_num_input_tokens_seen = mode
        trainer.processing_class = SimpleNamespace(pad_token_id=0)
        trainer.accelerator.gather = lambda value: value.repeat(6)
        trainer.state.num_input_tokens_seen = 17
        inputs = {'input_ids': torch.tensor([[1, 0, 2, 0]]), 'labels': torch.full((1, 4), -100)}
        trainer._track_num_input_tokens(inputs)
        trainer._track_num_input_tokens(inputs)
        assert trainer.state.num_input_tokens_seen == 17 + 2 * expected
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_legacy_public_metrics_delegate_to_hf(framework, tmp_path):
    trainer = make(framework, Engine(tp_size=1), tmp_path)
    try:
        assert not trainer._tp_sft_enabled
        assert trainer.get_total_train_batch_size(trainer.args) == Trainer.get_total_train_batch_size(trainer, trainer.args)
        sampler = DPBlockDistributedSampler(list(range(13)), num_replicas=12, rank=0, dp_size=2, dp_rank=0, rank_block=6, shuffle=False, drop_last=True)
        loader = EpochDrivenDataLoader(sampler.dataset, sampler=sampler)
        assert trainer.num_examples(loader) == 13
        trainer.args.include_num_input_tokens_seen = 'all'
        trainer.accelerator.gather = lambda value: value.repeat(6)
        trainer.state.num_input_tokens_seen = 17
        trainer._track_num_input_tokens({'input_ids': torch.tensor([[1, 2]])})
        assert trainer.state.num_input_tokens_seen == 29
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_actual_adapter_hf_train_loop_short_final_ga_and_reporting(
    framework, tmp_path, monkeypatch,
):
    from tests.engine.backends.fsdp2.test_tp_loss import PackedModel, batch, make_engine
    from torchturbo.runtime.cpu import CpuRuntime

    engine = make_engine()
    engine.config.gbs = 3
    engine.backend.fsdp_config = engine.config.backend_config
    engine.backend._device_runtime = CpuRuntime()
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda group=None: 1)
    engine.get_tp_info = lambda: {'tp_size': 2}
    engine.model_name_or_path = str(tmp_path / 'source_model')
    collected = []
    engine.collect_metrics = collected.append
    engine.report_metrics = lambda logs, gather: None
    # A finite loader exposes HF's shortened final accumulation window. Sampler
    # padding and distributed normalization have separate integration coverage.
    engine.create_dataloader = lambda dataset, **kwargs: torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=kwargs['collate_fn'],
    )
    steps = []
    reports = []
    compute_prepared_loss = engine.compute_prepared_loss

    def record_prepared_loss(model, inputs, *args, **kwargs):
        result = compute_prepared_loss(model, inputs, *args, **kwargs)
        steps.append((kwargs['gradient_accumulation_steps'], kwargs['num_items_in_batch']))
        reports.append(float(result[1].main_loss))
        return result

    engine.compute_prepared_loss = record_prepared_loss
    source = [batch(0, i % 3) for i in range(5)]
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[],
        per_device_train_batch_size=1, num_train_epochs=1, save_strategy='no',
        logging_strategy='no', remove_unused_columns=False, disable_tqdm=True,
        max_grad_norm=0,
    )
    args.train_dataloader_shuffle = False
    cls = make_adapter_class(framework, engine)
    try:
        trainer = cls(
            model=PackedModel(), args=args, train_dataset=source,
            data_collator=lambda values: values[0],
        )
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0)
        engine.optimizer = trainer.optimizer
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1)
        # The fixture keeps HF's scheduler because optional framework scheduler
        # imports are unrelated to the adapter loss and reporting hooks.
        trainer.create_scheduler = lambda *args, **kwargs: trainer.lr_scheduler
        trainer.accelerator.backward = lambda loss, **kwargs: loss.backward()
        result = trainer.train()
        assert trainer.state.global_step == 2
        assert steps == [(3, 6), (3, 6), (3, 6), (2, 3), (2, 3)]
        assert trainer.model.prepare_calls == 5
        assert collected == [trainer.model] * 5
        logits = trainer.model(input_ids=source[0]['input_ids']).logits
        reference = torch.nn.functional.cross_entropy(
            logits[0, [1, 3, 4]], torch.tensor([3, 4, 6]),
        )
        assert sum(reports) / 2 == pytest.approx(float(reference.detach()))
        assert result.training_loss == pytest.approx(float(reference.detach()))
    finally:
        engine_registry.clear()


def _registered_legacy_engine(monkeypatch):
    from tests.engine.test_core import DummyBackend
    from torchturbo.engine import backend_registry
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.runtime.cpu import CpuRuntime

    monkeypatch.setitem(backend_registry._BACKENDS, 'fsdp2', DummyBackend)
    monkeypatch.setattr('torchturbo.engine.core.get_device_runtime', CpuRuntime)
    return TorchTurboEngine()


def test_registered_legacy_backend_getter_and_target_capability(monkeypatch):
    from torchturbo.engine.backends.base import BaseBackend

    engine = _registered_legacy_engine(monkeypatch)
    assert isinstance(engine.backend, BaseBackend)
    assert engine.get_tp_info() == {'tp_size': 1, 'tp_rank': 0, 'tp_group': None}
    with pytest.raises(NotImplementedError, match='DummyBackend.*prepare_loss_targets'):
        engine.backend.prepare_loss_targets(
            torch.tensor([[1, 2, 3]]), torch.tensor([0, 3]), shift=1,
            ignore_index=-100,
        )


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_registered_legacy_backend_adapter_constructor_preserves_hf_path(
    monkeypatch, framework, tmp_path,
):
    engine = _registered_legacy_engine(monkeypatch)
    if framework == 'lf':
        cls = make_torchturbo_trainer(Trainer, engine=engine, stage='sft')
    else:
        engine_registry.set_engine(engine, stage='sft')
        cls = type('LegacySwiftHFTrainer', (TurboTrainerMixin, Trainer), {})
    model = torch.nn.Linear(1, 1, bias=False)
    model.is_diffusion_model = False
    try:
        trainer = cls(model=model, args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            gradient_accumulation_steps=3,
        ))
        assert not trainer._tp_sft_enabled
        assert trainer.args.gradient_accumulation_steps == 3
        assert not trainer.model_accepts_loss_kwargs
        assert trainer.get_total_train_batch_size(trainer.args) == (
            Trainer.get_total_train_batch_size(trainer, trainer.args)
        )
        batches = [{'input_ids': torch.tensor([[1, 2, 3]]),
                    'labels': torch.tensor([[-100, 2, 3]])}]
        assert trainer._get_num_items_in_batch(batches, torch.device('cpu')) is None
        assert 'labels' in batches[0]
    finally:
        engine_registry.clear()


@pytest.mark.parametrize('framework', ['lf', 'swift'])
def test_each_tp_training_step_requires_a_fresh_main_report(framework, tmp_path, monkeypatch):
    engine = Engine()
    trainer = make(framework, engine, tmp_path)
    try:
        samples, count = trainer.get_batch_samples(iter([
            {'input_ids': torch.tensor([[1, 2, 3]]), 'labels': torch.tensor([[-100, 2, 3]])},
        ]), 1, torch.device('cpu'))
        first = trainer.training_step(trainer.model, samples[0], num_items_in_batch=count)
        assert not first.requires_grad
        assert trainer._tp_main_loss is not None
        # Enforce the boundary for any superclass path bypassing compute_loss;
        # this controlled branch is not evidence of a framework production bug.
        monkeypatch.setattr(Trainer, 'training_step', lambda *args, **kwargs: torch.tensor(1.0))
        with pytest.raises(RuntimeError, match='TP.*main.*report'):
            trainer.training_step(trainer.model, samples[0], num_items_in_batch=count)
        assert trainer._tp_main_loss is None
    finally:
        engine_registry.clear()
