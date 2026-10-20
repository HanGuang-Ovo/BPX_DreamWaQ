"""DreamWaQActorCritic 单元测试：维度、接口兼容与特征复用（测试 1/4）。"""

import sys
import unittest

import torch

sys.path.insert(0, "src")

from bpx_dreamwaq.tasks.agents.dreamwaq_actor_critic import (
    DeployableDreamWaQPolicy,
    DreamWaQActorCritic,
)
from bpx_dreamwaq.tasks.obs_contract import (
    ACTION_DIM,
    DEFAULT_HISTORY_LENGTH,
    history_obs_dim,
    single_obs_dim,
)


def make_policy(**overrides):
    params = dict(
        num_actor_obs=single_obs_dim(),
        num_critic_obs=71,
        num_actions=ACTION_DIM,
        history_length=DEFAULT_HISTORY_LENGTH,
        cenet_encoder_hidden_dims=[32, 16],
        cenet_vel_head_hidden_dims=[16],
        cenet_vae_head_hidden_dims=[16],
        cenet_decoder_hidden_dims=[16, 16],
        cenet_latent_dim=16,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
    )
    params.update(overrides)
    return DreamWaQActorCritic(**params)


class TestDreamWaQActorCritic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = make_policy()
        self.batch = 6
        self.obs = torch.randn(self.batch, single_obs_dim())
        self.critic_obs = torch.randn(self.batch, 71)
        self.history = torch.randn(self.batch, history_obs_dim(DEFAULT_HISTORY_LENGTH))

    def test_actor_input_dimension(self):
        # 45 单帧 + 3 估计速度 + 16 隐变量 = 64。
        self.assertEqual(
            self.policy.actor[0].in_features, single_obs_dim() + 3 + self.policy.cenet_latent_dim
        )

    def test_act_and_action_dim(self):
        actions = self.policy.act(self.obs, history_obs=self.history)
        self.assertEqual(actions.shape, (self.batch, ACTION_DIM))
        log_prob = self.policy.get_actions_log_prob(actions)
        self.assertEqual(log_prob.shape, (self.batch,))

    def test_evaluate_critic(self):
        value = self.policy.evaluate(self.critic_obs)
        self.assertEqual(value.shape, (self.batch, 1))
        self.assertEqual(self.policy.critic[0].in_features, 71)

    def test_act_inference_deterministic(self):
        a1 = self.policy.act_inference(self.obs, history_obs=self.history)
        a2 = self.policy.act_inference(self.obs, history_obs=self.history)
        self.assertEqual(a1.shape, (self.batch, ACTION_DIM))
        self.assertTrue(torch.allclose(a1, a2))

    def test_features_detached_from_cenet(self):
        features = self.policy.encode_cenet(self.history)
        self.assertFalse(features.requires_grad)
        self.assertEqual(features.shape, (self.batch, 3 + self.policy.cenet_latent_dim))

    def test_act_with_cenet_features_reuses_stored_values(self):
        stored = torch.randn(self.batch, 3 + self.policy.cenet_latent_dim)
        self.policy.act_with_cenet_features(self.obs, stored)
        self.assertTrue(torch.allclose(self.policy._cenet_features, stored))

    def test_missing_features_raises(self):
        fresh = make_policy()
        with self.assertRaises(RuntimeError):
            fresh.update_distribution(self.obs)

    def test_checkpoint_roundtrip_includes_cenet(self):
        state = self.policy.state_dict()
        cenet_keys = [key for key in state if key.startswith("cenet.")]
        self.assertGreater(len(cenet_keys), 0)
        restored = make_policy()
        restored.load_state_dict(state)
        for key in cenet_keys[:5]:
            self.assertTrue(torch.allclose(state[key], restored.state_dict()[key]))

    def test_deployable_policy_shapes(self):
        deployable = DeployableDreamWaQPolicy(self.policy)
        deployable.eval()
        history = torch.randn(3, history_obs_dim(DEFAULT_HISTORY_LENGTH))
        with torch.no_grad():
            actions = deployable(history)
        self.assertEqual(actions.shape, (3, ACTION_DIM))
        # 最新帧提取：修改历史中最新帧必须影响输出。
        with torch.no_grad():
            base = deployable(history)
            modified = history.clone()
            modified[:, -ACTION_DIM:] += 0.5  # last_action 项的最新帧
            changed = deployable(modified)
        self.assertFalse(torch.allclose(base, changed))


if __name__ == "__main__":
    unittest.main()
