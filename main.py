import copy

import gym
import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.data.wrappers import RenderImageInfoWrapper
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


# Based on example from https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html
def main():
    rng = np.random.default_rng(0)

    env = gym.make("Pendulum-v1")
    env = RenderImageInfoWrapper(env)
    venv = DummyVecEnv(env_fns=[lambda: copy.deepcopy(env)])

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm,
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, rng=rng)

    querent = preference_comparisons.PrefCollectQuerent(
        pref_collect_address="http://127.0.0.1:8000",
        video_output_dir="/path/to/videofiles/",
        video_fps=20,
    )
    gatherer = preference_comparisons.PrefCollectGatherer(pref_collect_address="http://127.0.0.1:8000")

    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=3,
        rng=rng,
    )

    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        n_steps=2048 // venv.num_envs,
    )

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        rng=rng,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_querent=querent,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        initial_epoch_multiplier=1,
    )
    pref_comparisons.train(total_timesteps=5_000, total_comparisons=200)

    reward, _ = evaluate_policy(agent.policy, venv, 10)
    print("Reward:", reward)


if __name__ == '__main__':
    main()
