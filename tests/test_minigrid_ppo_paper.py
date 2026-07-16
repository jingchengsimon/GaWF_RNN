"""Tests for the isolated paper-protocol MiniGrid PPO implementation."""

from __future__ import annotations

import torch

from train_minigrid_ppo_paper import build_arg_parser, paper_learning_rate
from utils.minigrid_ppo_paper_models import (
    GaWFPolicyState,
    PaperMiniGridActorCritic,
    PaperMiniGridEncoder,
)
from utils.recurrent_cores.paper_lstm import PaperLSTMCore


def _observations(batch_size: int = 2, steps: int = 4) -> torch.Tensor:
    obs = torch.zeros(batch_size, steps, 3, 3, 3, dtype=torch.uint8)
    obs[:, :, 0].random_(0, 11)
    obs[:, :, 1].random_(0, 6)
    obs[:, :, 2].random_(0, 3)
    return obs


def test_paper_defaults_match_ppo2_protocol() -> None:
    args = build_arg_parser().parse_args([])
    assert args.total_timesteps == 100_000_000
    assert args.agent_view_size == 3
    assert args.num_steps == 128
    assert args.num_minibatches == 8
    assert args.update_epochs == 4
    assert args.gae_lambda == 0.95
    assert args.ent_coef == 0.01
    assert args.adam_eps == 1e-5


def test_paper_learning_rate_is_baseline_specific() -> None:
    assert paper_learning_rate("MiniGrid-MemoryS7-v0", "paper_lstm") == 1e-3
    assert paper_learning_rate("MiniGrid-MemoryS7-v0", "gawf") == 1e-5
    assert paper_learning_rate("MiniGrid-RedBlueDoors-8x8-v0", "paper_lstm") == 1e-5


def test_paper_encoder_is_exactly_five_tanh_layers() -> None:
    encoder = PaperMiniGridEncoder(grid_size=3)
    assert len(encoder.layers) == 5
    output = encoder(_observations(3, 1)[:, 0])
    assert output.shape == (3, 128)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)


def test_paper_lstm_has_no_post_recurrent_norm_or_activation() -> None:
    core = PaperLSTMCore(input_size=128, hidden_size=128)
    assert isinstance(core.rnn, torch.nn.LSTM)
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in core.modules())
    output, state = core(torch.randn(2, 5, 128))
    assert output.shape == (2, 5, 128)
    assert state[0].shape == (1, 2, 128)


def test_gawf_uses_action_logits_as_seven_dimensional_feedback() -> None:
    model = PaperMiniGridActorCritic(
        num_actions=7,
        grid_size=3,
        model_type="gawf",
        hidden_size=24,
    )
    state = model.initial_state(batch_size=2, device="cpu")
    assert isinstance(state, GaWFPolicyState)
    assert model.feedback_mode == "policy_logits"
    assert model.feedback_dim == 7
    assert state.prev_policy_logits.shape == (2, 7)

    with torch.no_grad():
        model.core.U.fill_(0.2)
        model.core.V.fill_(0.2)
    obs = _observations(2, 1)[:, 0]
    done = torch.zeros(2)
    state_a = GaWFPolicyState(state.recurrent, torch.zeros(2, 7))
    state_b = GaWFPolicyState(state.recurrent, torch.ones(2, 7))
    logits_a, _value_a, _next_a = model.step(obs, done, state_a)
    logits_b, _value_b, _next_b = model.step(obs, done, state_b)
    assert not torch.allclose(logits_a, logits_b)


def test_gawf_action_feedback_produces_uv_gradients() -> None:
    torch.manual_seed(7)
    model = PaperMiniGridActorCritic(
        num_actions=7,
        grid_size=3,
        model_type="gawf",
        hidden_size=24,
    )
    obs = _observations(3, 5)
    dones = torch.zeros(3, 5)
    actions = torch.randint(0, 7, (3, 5))
    logprobs, entropy, values = model.evaluate_actions_sequence(obs, dones, actions)
    loss = -(logprobs.mean() + 0.01 * entropy.mean()) + values.pow(2).mean()
    loss.backward()
    assert model.core.U.grad is not None
    assert model.core.V.grad is not None
    assert float(model.core.U.grad.abs().sum()) > 0.0
    assert float(model.core.V.grad.abs().sum()) > 0.0


def test_recurrent_state_minibatch_selection_preserves_layout() -> None:
    indices = torch.tensor([1, 3])
    for model_type in ("paper_lstm", "lstm_core", "rnn", "gru", "gawf"):
        model = PaperMiniGridActorCritic(
            num_actions=7,
            grid_size=3,
            model_type=model_type,
            hidden_size=16,
        )
        state = model.initial_state(batch_size=4, device="cpu")
        selected = model.select_state(state, indices)
        if isinstance(selected, GaWFPolicyState):
            assert selected.recurrent.shape[0] == 2
            assert selected.prev_policy_logits.shape == (2, 7)
        elif isinstance(selected, tuple):
            assert selected[0].shape == (1, 2, 16)
        else:
            assert selected.shape == (1, 2, 16)
