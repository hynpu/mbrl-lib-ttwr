import numpy as np

# map limits
x_min = 0
x_max = 60
y_min = -25
y_max = 25

map_margin = 5

map_x_min = x_min - map_margin
map_x_max = x_max + map_margin
map_y_min = y_min - map_margin
map_y_max = y_max + map_margin

# goal state is on the origin, x axis heading
goal_state = np.array([0, 0, 0, 0], dtype=np.float32)
dist_tolerance = 3
theta_tolerance = 0.2
phi_tolerance = 0.2

distance_reward_range = (1, 10)  # Desired range for distance reward

v1_min= -5 # max truck speed
v1_max= 2 # max truck speed

# host control limits
maxSteeringAngle = np.pi / 6 # 30 degree
jackKnifeAngle = np.deg2rad(30) # np.pi / 6 

# ttwr system parameters
L1 = 3
L2 = 1.5 #1.5 3
L3 = 2 #2 4
dt = 0.1


# host dimensions
host_length = 4.7
host_width = 1.8

# trailer dimensions
trailer_length = 2.2 # 2.2 5.2
trailer_width = 1.3
trailer_rear_overhang = 1.0 #1.0 5.0
trailer_front_overhang = trailer_length - trailer_rear_overhang

# wheel dimensions
wheel_radius = 0.4

# path resolution
path_res = 0.5

