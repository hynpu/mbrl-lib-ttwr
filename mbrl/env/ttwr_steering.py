import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import matplotlib.pyplot as plt

import mbrl.env.ttwr_assets.ttwr_config as ttwr_config
import mbrl.env.ttwr_assets.ttwr_helpers as ttwr_helpers

import os
import pygame

class TtwrEnv(gym.Env):
    # This is a continuous version of gym's cartpole environment, with the only difference
    # being valid actions are any numbers in the range [-1, 1], and the are applied as
    # a multiplicative factor to the total force.
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None):

        self.L1 = ttwr_config.L1
        self.L2 = ttwr_config.L2
        self.L3 = ttwr_config.L3
        self.dt = ttwr_config.dt

        # init input TODO: use velo as input instead of fixed value
        self.v1 = -2
        self.delta = 0

        self.steps = 0
        self.steps_beyond_terminated = None

        # minimal states required: [x2, y2, theta2, phi]
        self.state = np.zeros(4)
        self.state_init = np.zeros(4)
        self.full_state = np.zeros(8)
        low_obs_state = np.array([ttwr_config.x_min, ttwr_config.y_min, -np.pi, -ttwr_config.jackKnifeAngle], dtype=np.float32)
        high_obs_state = np.array([ttwr_config.x_max, ttwr_config.y_max, np.pi, ttwr_config.jackKnifeAngle], dtype=np.float32)

        # input action is steering angle
        act_min = np.array((-ttwr_config.maxSteeringAngle,), dtype=np.float32)
        act_max = np.array((ttwr_config.maxSteeringAngle,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low_obs_state, high_obs_state, dtype=np.float32)
        self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)

        self.reset()

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    
    # reset the trailer state, and compute the TTWR full states; also accept given trailer states
    def reset(self, seed: Optional[int] = None, desired_state: Optional[np.ndarray] = None, desired_v1 = None):
        super().reset(seed=seed)
        easy_mode = True
        if desired_state is not None:
            self.state = desired_state           
        else:
            if easy_mode:
                x2 = np.random.uniform(39, 41)
                y2 = np.random.uniform(19, 21)
                theta2 = np.pi/4 + np.random.uniform(-np.pi/10, np.pi/10)
                phi = 0 # np.random.uniform(-np.pi/10, np.pi/10)
                self.state = np.array([x2, y2, theta2, phi])
            else:
                distance = np.random.uniform(20, 40)
                heading = np.random.uniform(-np.pi/6, np.pi/6)
                x2 = distance * np.cos(heading)
                y2 = distance * np.sin(heading)
                theta2 = heading + np.random.uniform(-np.pi/6, np.pi/6)
                phi = 0 # np.random.uniform(-np.pi/10, np.pi/10)
                self.state = np.array([x2, y2, theta2, phi])

        self.state_init = self.state.copy()
        
        self.distance_init_to_goal = np.linalg.norm([self.state_init[0] - ttwr_config.goal_state[0], self.state_init[1] - ttwr_config.goal_state[1]])
        self.max_steps_allowed = self.distance_init_to_goal * 2 / 1.0 / ttwr_config.dt # assuming vehicle run 1m/s, 2 times distance to goal

        # if truck velocity is given by user
        if desired_v1 is not None:
            self.v1 = desired_v1
        
        self.steps = 0
        self.steps_beyond_terminated = None
        
        # trailer states from env
        self.compute_full_state(self.state)

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    # given trailer states, get TTWR states
    def compute_full_state(self, trailerState=np.zeros(4)):
        x2, y2, theta2, phi = trailerState

        theta1 = theta2 - phi
        x1 = x2 + self.L2 * np.cos(theta1) + self.L3 * np.cos(theta2)
        y1 = y2 + self.L2 * np.sin(theta1) + self.L3 * np.sin(theta2)

        self.full_state = np.array([x1, y1, theta1, x2, y2, theta2, phi, self.v1])

    def step(self, action):
        action = action.squeeze()
        # delta is the front wheel steering angle
        self.delta = action

        x2, y2, theta2, phi = self.state
        
        # trailer vehicle state update
        x2_dot = self.v1 * np.cos(phi) * (1 - self.L2 / self.L1 * np.tan(phi) * np.tan(self.delta)) * np.cos(theta2)
        y2_dot = self.v1 * np.cos(phi) * (1 - self.L2 / self.L1 * np.tan(phi) * np.tan(self.delta)) * np.sin(theta2)
        theta2_dot = -self.v1 * (np.sin(phi) / self.L3 + self.L2 / (self.L1 * self.L3) * np.cos(phi) * np.tan(self.delta))

        # host trailer angle update
        # phi = theta2 - theta1
        phi_dot = -self.v1 * (np.sin(phi) / self.L3 + self.L2 / (self.L1 * self.L3) * np.cos(phi) * np.tan(self.delta)) - self.v1 * np.tan(self.delta) / self.L1

        # ttwr states
        x2 += x2_dot * self.dt
        y2 += y2_dot * self.dt
        theta2 = ttwr_helpers.wrapToPi(theta2_dot * self.dt + theta2)
        phi += phi_dot * self.dt

        self.state = np.array([x2, y2, theta2, phi])

        # get ttwr based on trailer states and phi
        self.compute_full_state(self.state)

        episode_failed, goal_reached = self.is_done()
        
        reward = self.compute_reward()
        terminate_episode = episode_failed or goal_reached

        if self.render_mode == "human":
            self.render()

        # leave truncated as False and Info as empty for now
        return self.state, reward, terminate_episode, False, {}
    
    def compute_reward(self):
        # Unpack the state components
        x2, y2, theta2, phi, = self.state
        goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state

        # Calculate the distance to the goal position
        # 45 is specialized for the ttwr environment whose start is 40 20
        distance_to_goal = np.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2) / 45

        # Fit a 2nd order polynomial curve between the current position and the goal position
        delta_x = goal_x2 - x2
        delta_y = goal_y2 - y2
        m = np.tan(goal_theta2)
        a2 = ((y2 - goal_y2) - m * (x2 - goal_x2)) / (x2 - goal_x2) ** 2
        a1 = m - 2 * (((y2 - goal_y2) - m * (x2 - goal_x2)) / (x2 - goal_x2) ** 2) * goal_x2
        a0 = goal_y2 - a1 * goal_x2 - a2 * goal_x2 ** 2

        # Calculate the tangent angle of the polynomial curve at the current position x2, y2
        tangent_angle = np.arctan(2 * a2 * x2 + a1)

        # Calculate the alignment between the trailer's orientation and the tangent angle
        theta_alignment = np.cos(theta2 - tangent_angle)

        # Calculate the alignment reward component
        theta_alignment_reward = np.exp(-np.power(theta_alignment - 1, 2) / (2 * 0.5 ** 2))

        # Calculate the modified distance reward component
        distance_reward = 1 / (1 + distance_to_goal) ** 4

        # phi reward
        phi_alignment = np.cos(phi - goal_phi)
        phi_alignment_reward = np.exp(-np.power(phi_alignment - 1, 2) / (2 * 0.5 ** 2))

        # Adjust the alignment reward based on the distance to the goal
        distance_threshold = 5  # Adjust this threshold as needed
        weight_theta = 0.8 if distance_to_goal < distance_threshold else 1.0
        weight_dist = 0.8 if distance_to_goal < distance_threshold else 0.5
        weight_phi = 0.8 if distance_to_goal < distance_threshold else 0.5

        # Combine the alignment reward and distance reward
        total_reward = weight_theta * theta_alignment_reward + weight_dist * distance_reward + 0.2 * weight_phi * phi_alignment_reward

        # Apply penalties for episode failure and goal reached
        episode_failed, goal_reached = self.is_done()
        total_reward = -100.0 + total_reward if episode_failed else total_reward
        total_reward = 200.0 + total_reward if goal_reached else total_reward

        return total_reward
    

    def is_done(self):
        # Unpack the state components
        x2, y2, theta2, phi = self.state[0], self.state[1], self.state[2], self.state[3]
        goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state

        # Check for jackknife angle violation
        jackknife_violation = np.abs(phi) > ttwr_config.jackKnifeAngle

        # Check if the trailer is out of range
        out_of_range = (x2 < ttwr_config.x_min) | (x2 > ttwr_config.x_max) | (y2 < ttwr_config.y_min) | (y2 > ttwr_config.y_max)
        out_of_range = out_of_range | (y2 < -5) | (y2 > 25) | (x2 > 45)

        # Terminate if trailer cannot reach goal after 2*distance_to_goal/v1_min
        time_exceeded = self.steps >= self.max_steps_allowed

        # Combine the termination conditions
        episode_failed = jackknife_violation | out_of_range | time_exceeded

        # Compute the position and angle differences
        pos_diff = np.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2)
        theta_diff = np.abs(theta2 - goal_theta2)
        phi_diff = np.abs(phi - goal_phi)

        # Check if the goal is reached
        goal_reached = (pos_diff < ttwr_config.dist_tolerance) & (theta_diff < ttwr_config.theta_tolerance) & (phi_diff < ttwr_config.phi_tolerance)

        return episode_failed, goal_reached
    

    def close(self):
        pass

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return

        if self.render_mode == "human":
            screen_width = 600 * 1.5
            screen_height = 400 * 1.5
        else:  # mode == "rgb_array"
            screen_width = 600
            screen_height = 400

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        # Rendering code starts here
        x1, y1, theta1, x2, y2, theta2 = self.full_state[:6]

        # Scale the coordinates to fit the screen
        scale = min(screen_width / (ttwr_config.map_x_max - ttwr_config.map_x_min),
                    screen_height / (ttwr_config.map_y_max - ttwr_config.map_y_min))
        offset_x = (screen_width - scale * (ttwr_config.map_x_max - ttwr_config.map_x_min)) / 2
        offset_y = (screen_height - scale * (ttwr_config.map_y_max - ttwr_config.map_y_min)) / 2

        def transform(x, y):
            screen_x = int(scale * (x - ttwr_config.map_x_min) + offset_x)
            screen_y = int(scale * (y - ttwr_config.map_y_min) + offset_y)
            return screen_x, screen_y
        
        # host vehicle centroid point
        x1_cent = x1 + ttwr_config.L1/2 * np.cos(theta1)
        y1_cent = y1 + ttwr_config.L1/2 * np.sin(theta1)

        # host vehicle front reference point
        x1_front = x1 + ttwr_config.L1 * np.cos(theta1)
        y1_front = y1 + ttwr_config.L1 * np.sin(theta1)

        # hitch point
        hitch_x = x1 - ttwr_config.L2 * np.cos(theta1)
        hitch_y = y1 - ttwr_config.L2 * np.sin(theta1)

        # front wheels of host vehicle
        # compute left front wheel point using x1_front and y1_front
        x1_lf = x1_front - ttwr_config.host_width/2 * np.sin(theta1)
        y1_lf = y1_front + ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_lf_frt = x1_lf + ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_lf_frt = y1_lf + ttwr_config.wheel_radius * np.sin(theta1+self.delta)
        x1_lf_rear = x1_lf - ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_lf_rear = y1_lf - ttwr_config.wheel_radius * np.sin(theta1+self.delta)

        # compute right front wheel point using x1_front and y1_front
        x1_rf = x1_front + ttwr_config.host_width/2 * np.sin(theta1)
        y1_rf = y1_front - ttwr_config.host_width/2 * np.cos(theta1)
        # compute right front wheel after delta turn and wheel dimension
        x1_rf_frt = x1_rf + ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_rf_frt = y1_rf + ttwr_config.wheel_radius * np.sin(theta1+self.delta)
        x1_rf_rear = x1_rf - ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_rf_rear = y1_rf - ttwr_config.wheel_radius * np.sin(theta1+self.delta)

        # rear wheels of host vehicle
        # compute left rear wheel point using x1_front and y1_front
        x1_lr = x1 - ttwr_config.host_width/2 * np.sin(theta1)
        y1_lr = y1 + ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_lr_frt = x1_lr + ttwr_config.wheel_radius * np.cos(theta1)
        y1_lr_frt = y1_lr + ttwr_config.wheel_radius * np.sin(theta1)
        x1_lr_rear = x1_lr - ttwr_config.wheel_radius * np.cos(theta1)
        y1_lr_rear = y1_lr - ttwr_config.wheel_radius * np.sin(theta1)

        # compute left rear wheel point using x1_front and y1_front
        x1_rr = x1 + ttwr_config.host_width/2 * np.sin(theta1)
        y1_rr = y1 - ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_rr_frt = x1_rr + ttwr_config.wheel_radius * np.cos(theta1)
        y1_rr_frt = y1_rr + ttwr_config.wheel_radius * np.sin(theta1)
        x1_rr_rear = x1_rr - ttwr_config.wheel_radius * np.cos(theta1)
        y1_rr_rear = y1_rr - ttwr_config.wheel_radius * np.sin(theta1)

        # wheels of trailer vehicle
        # compute left trailer wheel point using x2 and y2
        x2_lt = x2 - ttwr_config.trailer_width/2 * np.sin(theta2)
        y2_lt = y2 + ttwr_config.trailer_width/2 * np.cos(theta2)
        # compute left front wheel after delta turn and wheel dimension
        x2_lt_frt = x2_lt + ttwr_config.wheel_radius * np.cos(theta2)
        y2_lt_frt = y2_lt + ttwr_config.wheel_radius * np.sin(theta2)
        x2_lt_rear = x2_lt - ttwr_config.wheel_radius * np.cos(theta2)
        y2_lt_rear = y2_lt - ttwr_config.wheel_radius * np.sin(theta2)
        # compute right trailer wheel point using x2 and y2
        x2_rt = x2 + ttwr_config.trailer_width/2 * np.sin(theta2)
        y2_rt = y2 - ttwr_config.trailer_width/2 * np.cos(theta2)
        # compute right front wheel after delta turn and wheel dimension
        x2_rt_frt = x2_rt + ttwr_config.wheel_radius * np.cos(theta2)
        y2_rt_frt = y2_rt + ttwr_config.wheel_radius * np.sin(theta2)
        x2_rt_rear = x2_rt - ttwr_config.wheel_radius * np.cos(theta2)
        y2_rt_rear = y2_rt - ttwr_config.wheel_radius * np.sin(theta2)

        # compute rectangle corner points of host vehicle
        host_x_rect = np.array([x1_cent + ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent + ttwr_config.host_length/2 * np.cos(theta1) - ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent - ttwr_config.host_length/2 * np.cos(theta1) - ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent - ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent + ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1)])
        host_y_rect = np.array([y1_cent + ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent + ttwr_config.host_length/2 * np.sin(theta1) + ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent - ttwr_config.host_length/2 * np.sin(theta1) + ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent - ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent + ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1)])

        # compute rectangle corner points of host vehicle
        trailer_x_rect = np.array([x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) - ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 - ttwr_config.trailer_rear_overhang * np.cos(theta2) - ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 - ttwr_config.trailer_rear_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2)])
        trailer_y_rect = np.array([y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) + ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 - ttwr_config.trailer_rear_overhang * np.sin(theta2) + ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 - ttwr_config.trailer_rear_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2)])

        # Transform the coordinates
        host_x_rect, host_y_rect = zip(*[transform(x, y) for x, y in zip(host_x_rect, host_y_rect)])
        trailer_x_rect, trailer_y_rect = zip(*[transform(x, y) for x, y in zip(trailer_x_rect, trailer_y_rect)])
        hitch_x, hitch_y = transform(hitch_x, hitch_y)
        x1, y1 = transform(x1, y1)
        x2, y2 = transform(x2, y2)
        x1_lf_frt, y1_lf_frt = transform(x1_lf_frt, y1_lf_frt)
        x1_lf_rear, y1_lf_rear = transform(x1_lf_rear, y1_lf_rear)
        x1_rf_frt, y1_rf_frt = transform(x1_rf_frt, y1_rf_frt)
        x1_rf_rear, y1_rf_rear = transform(x1_rf_rear, y1_rf_rear)
        x1_lr_frt, y1_lr_frt = transform(x1_lr_frt, y1_lr_frt)
        x1_lr_rear, y1_lr_rear = transform(x1_lr_rear, y1_lr_rear)
        x1_rr_frt, y1_rr_frt = transform(x1_rr_frt, y1_rr_frt)
        x1_rr_rear, y1_rr_rear = transform(x1_rr_rear, y1_rr_rear)
        x2_lt_frt, y2_lt_frt = transform(x2_lt_frt, y2_lt_frt)
        x2_lt_rear, y2_lt_rear = transform(x2_lt_rear, y2_lt_rear)
        x2_rt_frt, y2_rt_frt = transform(x2_rt_frt, y2_rt_frt)
        x2_rt_rear, y2_rt_rear = transform(x2_rt_rear, y2_rt_rear)

        # Draw host vehicle rectangle
        pygame.draw.polygon(self.surf, (40, 120, 181), list(zip(host_x_rect, host_y_rect)), 2)
        # Draw trailer vehicle rectangle
        pygame.draw.polygon(self.surf, (40, 120, 181), list(zip(trailer_x_rect, trailer_y_rect)), 2)
        # Draw host hitch point
        pygame.draw.circle(self.surf, (250, 127, 111), (hitch_x, hitch_y), 2)
        # Draw a line from hitch to host centroid
        pygame.draw.line(self.surf, (231, 218, 210), (hitch_x, hitch_y), (x1, y1), 2)
        # Draw a line from hitch to trailer centroid
        pygame.draw.line(self.surf, (231, 218, 210), (hitch_x, hitch_y), (x2, y2), 2)
        # Draw the host and trailer wheels
        pygame.draw.line(self.surf, (130, 176, 210), (x1_lf_frt, y1_lf_frt), (x1_lf_rear, y1_lf_rear), 3)  # host left front wheel
        pygame.draw.line(self.surf, (130, 176, 210), (x1_rf_frt, y1_rf_frt), (x1_rf_rear, y1_rf_rear), 3)  # host right front wheel
        pygame.draw.line(self.surf, (130, 176, 210), (x1_lr_frt, y1_lr_frt), (x1_lr_rear, y1_lr_rear), 3)  # host left rear wheel
        pygame.draw.line(self.surf, (130, 176, 210), (x1_rr_frt, y1_rr_frt), (x1_rr_rear, y1_rr_rear), 3)  # host right rear wheel
        pygame.draw.line(self.surf, (130, 176, 210), (x2_lt_frt, y2_lt_frt), (x2_lt_rear, y2_lt_rear), 3)  # trailer left wheel
        pygame.draw.line(self.surf, (130, 176, 210), (x2_rt_frt, y2_rt_frt), (x2_rt_rear, y2_rt_rear), 3)  # trailer right wheel

        # Draw the goal state
        pygame.draw.circle(self.surf, (255, 0, 0), transform(0, 0), 5)

        # Rendering code ends here

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


        def close(self):
            if self.screen is not None:
                pygame.display.quit()
                pygame.quit()
                self.isopen = False