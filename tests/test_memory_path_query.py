"""M1 regression tests. Run with: python -m pytest -q tests/test_memory_path_query.py."""

import copy

import pytest
import torch

from rsl_rl.modules import ActorCriticSRU
from rsl_rl.networks.sru_memory import PathEncoder
from rsl_rl.utils import split_and_pad_trajectories, unpad_trajectories


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous_threads)


def make_policy(mode=None, cameras=1):
    return ActorCriticSRU(
        num_actor_obs=6 + 12 + 16 * cameras,
        num_critic_obs=6 + 12 + 16 * cameras + 16 + 1,
        num_actions=3,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        image_input_dims=(8, 1, 2),
        height_input_dims=(8, 1, 2),
        rnn_hidden_size=16,
        path_length=12,
        path_embed_dim=8,
        dropout=0,
        **({"path_query_mode": mode} if mode is not None else {}),
    )


def assert_finite(tensor):
    assert torch.isfinite(tensor).all()


def assert_projection_grad(policy):
    grad = policy.actor_memory_to_path_query.weight.grad
    assert grad is not None
    assert_finite(grad)
    assert grad.abs().sum() > 0


def test_path_encoder_optional_query():
    encoder = PathEncoder(input_dim=4, d_model=8, max_len=3)
    path = torch.randn(3, 3, 4)
    original = encoder(path)
    query = encoder.latent_query.expand(3, -1)
    torch.testing.assert_close(encoder(path, None), original, rtol=0, atol=0)
    torch.testing.assert_close(encoder(path, query), original, rtol=0, atol=0)
    external = torch.randn(3, 8, requires_grad=True)
    output = encoder(path, external)
    assert_finite(output)
    assert not torch.allclose(output, original)
    output.square().mean().backward()
    assert external.grad is not None
    assert_finite(external.grad)
    assert encoder.latent_query.grad is None


@pytest.mark.parametrize("cameras", [1, 2])
def test_default_learned_compatibility_and_critic(cameras):
    torch.manual_seed(23)
    default = make_policy(cameras=cameras).eval()
    torch.manual_seed(23)
    explicit = make_policy("learned", cameras).eval()
    assert default.path_query_mode == "learned"
    assert default.actor_memory_to_path_query is None
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    memory = make_policy("memory", cameras).eval()
    result = memory.load_state_dict(default.state_dict(), strict=False)
    assert set(result.missing_keys) == {
        "actor_memory_to_path_query.weight", "actor_memory_to_path_query.bias"
    }
    assert not result.unexpected_keys
    for _ in range(2):
        obs = torch.randn(3, default.num_actor_obs)
        critic_obs = torch.randn(3, default.num_critic_obs)
        torch.testing.assert_close(default.act_inference(obs), explicit.act_inference(obs), rtol=0, atol=0)
        value = default.evaluate(critic_obs)
        torch.testing.assert_close(value, memory.evaluate(critic_obs), rtol=0, atol=0)
        assert_finite(value)


@pytest.mark.parametrize("cameras", [1, 2])
def test_memory_inference_backward_and_optimizer(cameras):
    from rsl_rl.algorithms import PPO

    policy = make_policy("memory", cameras)
    actions = policy.act_inference(torch.randn(3, policy.num_actor_obs))
    assert actions.shape == (3, 3)
    assert policy.linear_dropout_actor.linear.in_features == 16 + 8
    assert_finite(actions)
    actions.square().mean().backward()
    assert_projection_grad(policy)
    assert policy.attn_path_net.latent_query.grad is None
    actor_ids = {id(p) for p in policy.get_actor_parameters()}
    critic_ids = {id(p) for p in policy.get_critic_parameters()}
    assert id(policy.attn_path_net.latent_query) not in actor_ids
    optimizer = PPO(policy).optimizer
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(policy.attn_path_net.latent_query) not in optimizer_ids
    for p in policy.actor_memory_to_path_query.parameters():
        assert id(p) in actor_ids
        assert id(p) in optimizer_ids
        assert id(p) not in critic_ids
    before = policy.actor_memory_to_path_query.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, policy.actor_memory_to_path_query.weight)


@pytest.mark.parametrize("mode", ["learned", "memory"])
@pytest.mark.parametrize("cameras", [1, 2])
def test_parameter_selection(mode, cameras):
    policy = make_policy(mode, cameras)
    latent = policy.attn_path_net.latent_query
    actor_ids = {id(p) for p in policy.get_actor_parameters()}
    assert (id(latent) in actor_ids) == (mode == "learned")
    assert latent.requires_grad
    assert dict(policy.named_parameters())["attn_path_net.latent_query"] is latent
    assert "attn_path_net.latent_query" in policy.state_dict()
    path_ids = {id(p) for p in policy.attn_path_net.parameters()}
    expected_path_ids = path_ids if mode == "learned" else path_ids - {id(latent)}
    assert actor_ids & path_ids == expected_path_ids
    if mode == "memory":
        assert {id(p) for p in policy.actor_memory_to_path_query.parameters()} <= actor_ids

    # The actual optimizer iterator excludes exactly the inactive Actor query.
    registered_ids = {id(parameter) for _, parameter in policy.named_parameters()}
    expected_ids = registered_ids if mode == "learned" else registered_ids - {id(latent)}
    assert {id(p) for p in policy.parameters()} == expected_ids
    assert [id(p) for p in policy.parameters(recurse=False)] == [id(policy.log_std)]

    # Critic membership is identical to its original module selection.
    critic_modules = [policy.attn_height_net, policy.attn_critic_image_net,
                      policy.attn_critic_path_net, policy.linear_dropout_critic,
                      policy.time_layer, policy.critic, policy.memory_c]
    expected_critic_ids = {id(p) for module in critic_modules for p in module.parameters()}
    assert {id(p) for p in policy.get_critic_parameters()} == expected_critic_ids
    assert id(policy.attn_critic_path_net.latent_query) in expected_critic_ids
    assert policy.attn_critic_path_net.latent_query.requires_grad


@pytest.mark.parametrize("mode", ["learned", "memory"])
def test_mdpo_muon_optimizer_step(mode):
    from rsl_rl.algorithms import MDPO

    policies = [make_policy(mode), make_policy(mode)]
    algorithm = MDPO(*policies, device="cpu", use_muon=True)
    for policy, optimizer in zip(policies, [algorithm.optimizer_1, algorithm.optimizer_2]):
        assert any(group.get("use_muon") for group in optimizer.param_groups)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert optimizer_ids == {id(p) for p in policy.get_actor_parameters() + policy.get_critic_parameters()}
        latent = policy.attn_path_net.latent_query
        assert (id(latent) in optimizer_ids) == (mode == "learned")
        policy.act(torch.randn(3, policy.num_actor_obs))
        values = policy.evaluate(torch.randn(3, policy.num_critic_obs))
        loss = policy.action_mean.square().mean() + values.square().mean() + policy.entropy.mean()
        loss.backward()
        if mode == "memory":
            assert_projection_grad(policy)
            assert latent.grad is None
            parameter = policy.actor_memory_to_path_query.weight
        else:
            assert latent.grad is not None
            assert_finite(latent.grad)
            parameter = latent
        before = parameter.detach().clone()
        optimizer.step()
        assert_finite(parameter)
        assert not torch.equal(before, parameter)


@pytest.mark.parametrize("mode", ["learned", "memory"])
@pytest.mark.parametrize("cameras", [1, 2])
@pytest.mark.parametrize("terminations", [False, True])
def test_recurrent_alignment_and_backward(mode, cameras, terminations):
    policy = make_policy(mode, cameras).eval()
    seq_len, num_envs = 5, 3
    rollout = torch.randn(seq_len, num_envs, policy.num_actor_obs)
    dones = torch.zeros(seq_len, num_envs, 1, dtype=torch.bool)
    if terminations:
        dones[1, 0] = True
        dones[0, 1] = True
        dones[3, 1] = True
        dones[2, 2] = True
    padded, masks = split_and_pad_trajectories(rollout, dones)
    padded = padded.detach().requires_grad_()
    if terminations:
        assert (~masks).any()
        assert masks.sum() < masks.numel()  # Padded L*B differs from N_valid.
    batch_size = padded.shape[1]
    hidden = tuple(torch.randn(1, batch_size, 16) for _ in range(2))

    # Independent step-by-step inference for each terminated trajectory, with
    # its own initial state, supplies a reference for timestep/path alignment.
    reference_policy = copy.deepcopy(policy)
    reference_padded = torch.zeros(seq_len, batch_size, 3)
    with torch.no_grad():
        for trajectory in range(batch_size):
            reference_policy.memory_a.hidden_states = tuple(
                state[:, trajectory:trajectory + 1].clone() for state in hidden
            )
            for timestep in range(int(masks[:, trajectory].sum())):
                reference_padded[timestep, trajectory] = reference_policy.act_inference(
                    padded[timestep, trajectory:trajectory + 1]
                )[0]
    expected = unpad_trajectories(reference_padded, masks)
    output = policy.act_inference(padded, masks=masks, hidden_states=hidden)
    assert output.shape == (seq_len, num_envs, 3)
    assert_finite(output)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    assert_finite(padded.grad)
    assert torch.count_nonzero(padded.grad[~masks]) == 0
    if mode == "memory":
        assert_projection_grad(policy)


@pytest.mark.parametrize("mode", ["learned", "memory"])
@pytest.mark.parametrize("cameras", [1, 2])
def test_jit_export_parity(tmp_path, mode, cameras):
    policy = make_policy(mode, cameras).eval()
    policy.export_jit(str(tmp_path))
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    with torch.no_grad():
        for _ in range(2):
            obs = torch.randn(1, policy.num_actor_obs)
            expected = policy.act_inference(obs)
            actual = exported(obs)
            assert_finite(actual)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        exported.reset()
        policy.memory_a.hidden_states = None
        torch.testing.assert_close(exported(obs), policy.act_inference(obs), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["learned", "memory"])
@pytest.mark.parametrize("cameras", [1, 2])
def test_onnx_export(tmp_path, mode, cameras):
    onnx = pytest.importorskip("onnx")
    policy = make_policy(mode, cameras).eval()
    policy.export_onnx(str(tmp_path))
    model = onnx.load(str(tmp_path / "policy.onnx"))
    onnx.checker.check_model(model, full_check=True)
    assert [item.name for item in model.graph.input] == ["obs", "h_in", "c_in"]
    assert [item.name for item in model.graph.output] == ["actions", "h_out", "c_out"]
    from onnx.reference import ReferenceEvaluator

    runtime = ReferenceEvaluator(model)
    # Exercise the declared dynamic batch axis, despite export's B=1 example.
    obs = torch.randn(3, policy.num_actor_obs)
    hidden = torch.zeros(1, 3, 16)
    cell = torch.zeros_like(hidden)
    with torch.no_grad():
        for _ in range(2):
            expected = policy.act_inference(obs)
            actions, h_out, c_out = runtime.run(None, {
                "obs": obs.numpy(), "h_in": hidden.numpy(), "c_in": cell.numpy(),
            })
            actual = torch.from_numpy(actions)
            assert_finite(actual)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            hidden, cell = torch.from_numpy(h_out), torch.from_numpy(c_out)


def test_invalid_query_mode():
    with pytest.raises(ValueError, match="path_query_mode"):
        make_policy("invalid")
