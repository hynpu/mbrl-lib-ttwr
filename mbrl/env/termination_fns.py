# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import mbrl.env.ttwr_assets.ttwr_config as ttwr_config
import numpy as np

def hopper(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(-1)
        * (next_obs[:, 1:].abs() < 100).all(-1)
        * (height > 0.7)
        * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done

def ttwrTermination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:    
    assert len(next_obs.shape) == 2
    
    # Unpack the state components
    x2, y2, theta2, phi = next_obs[:, 0], next_obs[:, 1], next_obs[:, 2], next_obs[:, 3]
    
    goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state
    
    # Check for jackknife angle violation
    jackknife_violation = torch.abs(phi) > ttwr_config.jackKnifeAngle
    
    # Check if the trailer is out of range
    out_of_range = (x2 < ttwr_config.x_min) | (x2 > ttwr_config.x_max) | (y2 < ttwr_config.y_min) | (y2 > ttwr_config.y_max)
    
    # Combine the termination conditions
    episode_failed = jackknife_violation | out_of_range
    
    # Compute the position and angle differences
    pos_diff = torch.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2)
    theta_diff = torch.abs(theta2 - goal_theta2)
    phi_diff = torch.abs(phi - goal_phi)
    
    goal_reached = (pos_diff < ttwr_config.dist_tolerance) & (theta_diff < ttwr_config.theta_tolerance) & (phi_diff < ttwr_config.phi_tolerance)
    
    return episode_failed, goal_reached


def ttwrParking(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2
    
    episode_failed, goal_reached = ttwrTermination(act, next_obs)

    done = episode_failed | goal_reached
    done = done[:, None]
    
    return done


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done


def walker2d(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def ant(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done


def humanoid(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    z = next_obs[:, 0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:, None]
    return done
