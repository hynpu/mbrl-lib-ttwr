import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import matplotlib.pyplot as plt

import mbrl.env.ttwr_assets.ttwr_config as ttwr_config
import mbrl.env.ttwr_assets.ttwr_helpers as ttwr_helpers


class TtwrEnv(gym.Env):
    # This is a continuous version of gym's cartpole environment, with the only difference
    # being valid actions are any numbers in the range [-1, 1], and the are applied as
    # a multiplicative factor to the total force.
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

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
        self.viewer = None
        if self.render_mode is not None:
            # Create figure and axes only if render_mode is not None
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            self.fig = None
            self.ax = None

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

        goal_reached, total_reward = self.is_parked()
        episode_fail, failure_reward = self.is_failed()
        
        reward = total_reward + failure_reward
        terminate_episode = goal_reached or episode_fail

        # leave truncated as False and Info as empty for now
        return self.state, reward, terminate_episode, False, {}
    
    def is_parked(self):
        # Unpack the state components
        x2, y2, theta2, phi, = self.state
        goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state

        # Compute the position and angle differences
        pos_diff = np.sqrt((x2 - goal_x2) ** 2 + (y2 - goal_y2) ** 2)
        theta_diff = np.abs(theta2 - goal_theta2)
        phi_diff = np.abs(phi - goal_phi)

        # Check if the agent is within the goal tolerance
        if (pos_diff < ttwr_config.dist_tolerance) & (theta_diff < ttwr_config.theta_tolerance) & (phi_diff < ttwr_config.phi_tolerance):
            goal_reached = True
            parking_reward = 100
        else:
            goal_reached = False
            parking_reward = 0
        
        # reward computation
        theta_reward = -theta_diff ** 2
        phi_reward = -phi_diff ** 2
        distance_reward = np.tanh(pos_diff / self.distance_init_to_goal) # normalized distance reward
        distance_reward = ttwr_config.distance_reward_range[0] + (ttwr_config.distance_reward_range[1] - ttwr_config.distance_reward_range[0]) * distance_reward

        total_reward = 0.5 * distance_reward + 0.25 * theta_reward + 0.25 * phi_reward + parking_reward

        return goal_reached, total_reward
    

    def is_failed(self):
        # Unpack the state components
        x2, y2, theta2, phi, = self.state
        goal_x2, goal_y2, goal_theta2, goal_phi = ttwr_config.goal_state

        # Check for jackknife angle violation
        jackknife_violation = np.abs(phi) > ttwr_config.jackKnifeAngle

        # Check if the trailer is out of range
        out_of_range = (x2 < ttwr_config.x_min) | (x2 > ttwr_config.x_max) | (y2 < ttwr_config.y_min) | (y2 > ttwr_config.y_max)

        # terminate if trailer cannot reach goal after 2*distance_to_goal/v1_min
        time_exceeded = self.steps >= self.max_steps_allowed

        # Combine the termination conditions
        episode_terminate = jackknife_violation | out_of_range | time_exceeded

        failure_reward = -100

        return episode_terminate, failure_reward

    def close(self):
        pass

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return
        
        x1, y1, theta1, x2, y2, theta2 = self.full_state[:6]

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

        self.ax.clear()
        
        # plot host vehicle rectangle
        self.ax.plot(host_x_rect, host_y_rect, 'g')
        # plot host vehicle rectangle
        self.ax.plot(trailer_x_rect, trailer_y_rect, 'g')
        # plot host hitch point
        self.ax.add_artist(plt.Circle((hitch_x, hitch_y), .25, fill=False))
        # plot a line from hitch to host centroid
        self.ax.plot([hitch_x, x1], [hitch_y, y1], 'k')
        # plot a line from hitch to trailer centroid
        self.ax.plot([hitch_x, x2], [hitch_y, y2], 'k')
        # plot the host and trailer wheels
        self.ax.plot([x1_lf_frt, x1_lf_rear], [y1_lf_frt, y1_lf_rear], 'k', linewidth=2) # host left front wheel
        self.ax.plot([x1_rf_frt, x1_rf_rear], [y1_rf_frt, y1_rf_rear], 'k', linewidth=2) # host right front wheel
        self.ax.plot([x1_lr_frt, x1_lr_rear], [y1_lr_frt, y1_lr_rear], 'k', linewidth=2) # host left rear wheel
        self.ax.plot([x1_rr_frt, x1_rr_rear], [y1_rr_frt, y1_rr_rear], 'k', linewidth=2) # host right rear wheel
        self.ax.plot([x2_lt_frt, x2_lt_rear], [y2_lt_frt, y2_lt_rear], 'k', linewidth=2) # trailer left wheel
        self.ax.plot([x2_rt_frt, x2_rt_rear], [y2_rt_frt, y2_rt_rear], 'k', linewidth=2) # trailer right wheel

        # ax.plot(cur_state[0], cur_state[1], 'o')
        # ax.plot(cur_state[3], cur_state[4], 'x')
        self.ax.axis('equal')
        self.ax.set(xlim=(ttwr_config.map_x_min, ttwr_config.map_x_max), ylim=(ttwr_config.map_y_min, ttwr_config.map_y_max))

        plt.pause(np.finfo(np.float32).eps)