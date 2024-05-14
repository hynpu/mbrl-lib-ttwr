# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np

from . import termination_fns
import mbrl.env.ttwr_assets.ttwr_config as ttwr_config

def ttwrParking(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    # Unpack the state components
    x2, y2, theta2, phi = next_obs[:, 0], next_obs[:, 1], next_obs[:, 2], next_obs[:, 3]
    goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state
    

    # Compute the position and angle differences
    pos_diff = torch.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2)
    theta_diff = torch.abs(theta2 - goal_theta2)
    phi_diff = torch.abs(phi - goal_phi)
    
    # Reward computation
    theta_reward = -theta_diff ** 2
    phi_reward = -phi_diff ** 2
    distance_reward = pos_diff  # normalized distance reward
    distance_reward = ttwr_config.distance_reward_range[0] + (ttwr_config.distance_reward_range[1] - ttwr_config.distance_reward_range[0]) * distance_reward
    
    total_reward = 0.5 * distance_reward + 0.25 * theta_reward + 0.25 * phi_reward

    episode_failed, goal_reached = termination_fns.ttwrTermination(act, next_obs)

    # for episode_failed, punish the agent
    total_reward = torch.where(episode_failed, torch.tensor(-100.0), total_reward)

    # for goal reached, reward the agent
    total_reward = torch.where(goal_reached, torch.tensor(100.0), total_reward)

    # reshape the reward tensor
    total_reward = total_reward[:, None]
    
    return total_reward


def ttwrParkingPolynomial(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    
    # Unpack the state components
    x2, y2, theta2, phi = next_obs[:, 0], next_obs[:, 1], next_obs[:, 2], next_obs[:, 3]
    goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state

    # Calculate the distance to the goal position
    # 45 is specialized for the ttwr environment whose start is 40 20
    distance_to_goal = torch.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2) / 45
    
    # Fit a 2nd order polynomial curve between the current position and the goal position
    delta_x = goal_x2 - x2
    delta_y = goal_y2 - y2
    
    m = np.tan(goal_theta2)
    a2 = ((y2 - goal_y2) - m * (x2 - goal_x2)) / (x2 - goal_x2) ** 2
    a1 = m - 2 * (((y2 - goal_y2) - m * (x2 - goal_x2)) / (x2 - goal_x2) ** 2) * goal_x2
    a0 = goal_y2 - a1 * goal_x2 - a2 * goal_x2 ** 2

    # Calculate the tangent angle of the polynomial curve at the current position x2, y2
    tangent_angle = torch.atan(2 * a2 * x2 + a1)
    
    # Calculate the alignment between the trailer's orientation and the tangent angle
    theta_alignment = torch.cos(theta2 - tangent_angle)
    
    # Calculate the alignment reward component
    theta_alignment_reward = torch.exp(-torch.pow(theta_alignment - 1, 2) / (2 * 0.5 ** 2))
    
    # Calculate the modified distance reward component
    distance_reward = 1 / (1 + distance_to_goal) ** 4

    # phi reward
    phi_alignment = torch.cos(phi - goal_phi)
    phi_alignment_reward = torch.exp(-torch.pow(phi_alignment - 1, 2) / (2 * 0.5 ** 2))
    
    # Adjust the alignment reward based on the distance to the goal
    distance_threshold = 5  # Adjust this threshold as needed
    weight_theta = torch.where(distance_to_goal < distance_threshold, torch.tensor(0.8), torch.tensor(1))
    weight_dist = torch.where(distance_to_goal < distance_threshold, torch.tensor(0.8), torch.tensor(0.5))
    weight_phi = torch.where(distance_to_goal < distance_threshold, torch.tensor(0.8), torch.tensor(0.5))
    
    # Combine the alignment reward and distance reward
    total_reward = weight_theta * theta_alignment_reward + weight_dist * distance_reward + 0.2 * weight_phi * phi_alignment_reward
    
    # Apply penalties for episode failure and goal reached
    episode_failed, goal_reached = termination_fns.ttwrTermination(act, next_obs)
    total_reward = torch.where(episode_failed, torch.tensor(-100.0)+total_reward, total_reward)
    total_reward = torch.where(goal_reached, torch.tensor(200.0)+total_reward, total_reward)
    
    # Reshape the reward tensor
    total_reward = total_reward[:, None]
    
    return total_reward


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)
