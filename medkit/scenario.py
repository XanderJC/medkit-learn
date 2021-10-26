import warnings

import gym
import numpy as np
import torch
from gym import spaces
from tqdm import tqdm


class scenario(gym.Env):
    def __init__(
        self,
        domain,
        environment,
        policy,
        confounders=None,
        overlooked=None,
        stochastic=False,
        variation=1.0,
    ):
        """
        Main object of medkit.
        Takes a domain, environemnt, and policy to build the scene.

        Inherits from gym.Env for simple simulation access to the environment which
        also provides decision support in terms of predicted actions from the policy.

        Args:
            domain (str, optional): Domain. Defaults to "Ward".
            environment (str, optional): Environment Model. Defaults to "SVAE".
            policy (str, optional): Policy Model. Defaults to "LSTM".
            confounders ([type], optional): List of confounding variables. Defaults to None.
            overlooked ([type], optional): List of ignored variables. Defaults to None.
            stochastic (bool, optional): Stochastically sampled actions. Defaults to False.
            variation (float, optional): Sampling temperature. Defaults to 1.0.
        """

        self.dom = domain
        self.env = environment
        self.pol = policy

        self.ol_mask = torch.ones(self.dom.out_dim).float()
        if overlooked is not None:
            ol_index = [self.dom.series_names.index(var) for var in overlooked]
            ol_mask = np.ones(self.dom.out_dim)
            ol_mask[ol_index] = 0
            self.ol_mask = torch.tensor(ol_mask).float()

        self.hc_index = list(range(self.dom.out_dim))
        if confounders is not None:
            not_hidden = [
                var for var in self.dom.series_names if (var not in confounders)
            ]
            self.hc_index = [self.dom.series_names.index(var) for var in not_hidden]
            self.series_names = not_hidden

        self.stochastic = stochastic
        self.temp = variation

    def batch_generate(self, num_trajs=10, max_seq_length=50):
        """
        Generate batch dataset.

        Args:
            num_trajs (int, optional): Number of trajectories. Defaults to 10.
            max_seq_length (int, optional): Max trajectory length. Defaults to 50.

        Returns:
            static_data, series_data, action_data: batch data set
        """

        static_data = np.zeros((num_trajs, self.dom.static_in_dim))
        series_data = np.zeros((num_trajs, max_seq_length, self.dom.series_in_dim))
        action_data = np.zeros((num_trajs, max_seq_length, 1))

        for i in tqdm(range(num_trajs), desc="Generating trajectories"):

            static_obs, observation, info = self.reset()

            done = False
            seq_length = 1
            while (not done) & (seq_length < max_seq_length):

                action = info["predicted_action"]
                observation, reward, info, done = self.step(action)
                seq_length += 1

            prev_obs, prev_acts = self.history

            static_data[i, :] = static_obs
            series_data[i, : seq_length - 1, :] = prev_obs[:, : seq_length - 1, :]
            action_data[i, : seq_length - 1, :] = prev_acts[:, : seq_length - 1, :]

        self.ol_mask = torch.ones(self.dom.out_dim).float()

        series_data = series_data[:, :, self.hc_index]

        return static_data, series_data, action_data

    def step(self, action):
        """
        Wraps self.env.step(action) to incorporate predicted actions from policy.
        Additionally records history of the trajectory.
        """
        observation, reward, info, done = self.env.step(action)

        prev_obs, prev_acts = self.history
        if prev_acts is not None:
            prev_acts = torch.cat((prev_acts, action.reshape(1, 1, 1)), 1)
        else:
            prev_acts = action.reshape((1, 1, 1))
        prev_obs = torch.cat(
            (prev_obs, observation.reshape(1, 1, self.dom.series_in_dim)), 1
        )
        self.history = (prev_obs, prev_acts)

        t = prev_obs.shape[1]
        ol_prev_obs = prev_obs * self.ol_mask.expand(1, t, self.dom.out_dim)
        history = (ol_prev_obs, prev_acts)

        pred_action = self.pol.select_action(history, self.stochastic, self.temp)
        info = {"predicted_action": pred_action}

        observation = observation[self.hc_index]

        return observation, reward, info, done

    def reset(self):
        """
        Wraps self.env.reset() to incorporate predicted actions from policy.
        Additionally records history of the trajectory.
        """
        static_obs, observation = self.env.reset()
        self.history = (observation.reshape(1, 1, self.dom.series_in_dim), None)

        pred_action = self.pol.select_action(self.history, self.stochastic, self.temp)
        info = {"predicted_action": pred_action}

        observation = observation[self.hc_index]

        return static_obs, observation, info
