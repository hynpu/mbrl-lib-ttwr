import numpy as np
import assets.ttwr as ttwr_conf

def wrapTo2Pi(angle):
    posIn = angle>0
    angle = angle % (2*np.pi)
    if angle == 0 and posIn:
        angle = 2*np.pi
    return angle

def wrapToPi(angle):
    q = (angle < -np.pi) or (np.pi < angle)
    if(q):
        angle = wrapTo2Pi(angle + np.pi) - np.pi
    return angle

import matplotlib.pyplot as plt
import numpy as np

class VehicleRender():
    def __init__(self, ttwr_conf, render_mode=None, xlim=0, ylim=0):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.ttwr_conf = ttwr_conf

def renderTtwr(self, figure, axes, full_state, delta):
    x1 = full_state[0]
    y1 = full_state[1]
    theta1 = full_state[2]
    x2 = full_state[3]
    y2 = full_state[4]
    theta2 = full_state[5]
    host_len = ttwr_conf.host_length
    host_wid = ttwr_conf.host_width
    trailer_front_overhang = ttwr_conf.trailer_front_overhang
    trailer_rear_overhang = ttwr_conf.trailer_rear_overhang
    trailer_width = ttwr_conf.trailer_width

    # host vehicle centroid point
    x1_cent = x1 + ttwr_conf.L1/2 * np.cos(theta1)
    y1_cent = y1 + ttwr_conf.L1/2 * np.sin(theta1)

    # host vehicle front reference point
    x1_front = x1 + ttwr_conf.L1 * np.cos(theta1)
    y1_front = y1 + ttwr_conf.L1 * np.sin(theta1)
    
    # hitch point
    hitch_x = x1 - ttwr_conf.L2 * np.cos(theta1)
    hitch_y = y1 - ttwr_conf.L2 * np.sin(theta1)

    # front wheels of host vehicle
    # compute left front wheel point using x1_front and y1_front
    x1_lf = x1_front - ttwr_conf.host_width/2 * np.sin(theta1)
    y1_lf = y1_front + ttwr_conf.host_width/2 * np.cos(theta1)
    # compute left front wheel after delta turn and wheel dimension
    x1_lf_frt = x1_lf + ttwr_conf.wheel_radius * np.cos(theta1+delta)
    y1_lf_frt = y1_lf + ttwr_conf.wheel_radius * np.sin(theta1+delta)
    x1_lf_rear = x1_lf - ttwr_conf.wheel_radius * np.cos(theta1+delta)
    y1_lf_rear = y1_lf - ttwr_conf.wheel_radius * np.sin(theta1+delta)

    # compute right front wheel point using x1_front and y1_front
    x1_rf = x1_front + ttwr_conf.host_width/2 * np.sin(theta1)
    y1_rf = y1_front - ttwr_conf.host_width/2 * np.cos(theta1)
    # compute right front wheel after delta turn and wheel dimension
    x1_rf_frt = x1_rf + ttwr_conf.wheel_radius * np.cos(theta1+delta)
    y1_rf_frt = y1_rf + ttwr_conf.wheel_radius * np.sin(theta1+delta)
    x1_rf_rear = x1_rf - ttwr_conf.wheel_radius * np.cos(theta1+delta)
    y1_rf_rear = y1_rf - ttwr_conf.wheel_radius * np.sin(theta1+delta)
    
    # rear wheels of host vehicle
    # compute left rear wheel point using x1_front and y1_front
    x1_lr = x1 - ttwr_conf.host_width/2 * np.sin(theta1)
    y1_lr = y1 + ttwr_conf.host_width/2 * np.cos(theta1)
    # compute left front wheel after delta turn and wheel dimension
    x1_lr_frt = x1_lr + ttwr_conf.wheel_radius * np.cos(theta1)
    y1_lr_frt = y1_lr + ttwr_conf.wheel_radius * np.sin(theta1)
    x1_lr_rear = x1_lr - ttwr_conf.wheel_radius * np.cos(theta1)
    y1_lr_rear = y1_lr - ttwr_conf.wheel_radius * np.sin(theta1)

    # compute left rear wheel point using x1_front and y1_front
    x1_rr = x1 + ttwr_conf.host_width/2 * np.sin(theta1)
    y1_rr = y1 - ttwr_conf.host_width/2 * np.cos(theta1)
    # compute left front wheel after delta turn and wheel dimension
    x1_rr_frt = x1_rr + ttwr_conf.wheel_radius * np.cos(theta1)
    y1_rr_frt = y1_rr + ttwr_conf.wheel_radius * np.sin(theta1)
    x1_rr_rear = x1_rr - ttwr_conf.wheel_radius * np.cos(theta1)
    y1_rr_rear = y1_rr - ttwr_conf.wheel_radius * np.sin(theta1)

    # wheels of trailer vehicle
    # compute left trailer wheel point using x2 and y2
    x2_lt = x2 - ttwr_conf.trailer_width/2 * np.sin(theta2)
    y2_lt = y2 + ttwr_conf.trailer_width/2 * np.cos(theta2)
    # compute left front wheel after delta turn and wheel dimension
    x2_lt_frt = x2_lt + ttwr_conf.wheel_radius * np.cos(theta2)
    y2_lt_frt = y2_lt + ttwr_conf.wheel_radius * np.sin(theta2)
    x2_lt_rear = x2_lt - ttwr_conf.wheel_radius * np.cos(theta2)
    y2_lt_rear = y2_lt - ttwr_conf.wheel_radius * np.sin(theta2)
    # compute right trailer wheel point using x2 and y2
    x2_rt = x2 + ttwr_conf.trailer_width/2 * np.sin(theta2)
    y2_rt = y2 - ttwr_conf.trailer_width/2 * np.cos(theta2)
    # compute right front wheel after delta turn and wheel dimension
    x2_rt_frt = x2_rt + ttwr_conf.wheel_radius * np.cos(theta2)
    y2_rt_frt = y2_rt + ttwr_conf.wheel_radius * np.sin(theta2)
    x2_rt_rear = x2_rt - ttwr_conf.wheel_radius * np.cos(theta2)
    y2_rt_rear = y2_rt - ttwr_conf.wheel_radius * np.sin(theta2)

    # compute rectangle corner points of host vehicle
    host_x_rect = np.array([x1_cent + host_len/2 * np.cos(theta1) + host_wid/2 * np.sin(theta1), \
                            x1_cent + host_len/2 * np.cos(theta1) - host_wid/2 * np.sin(theta1), \
                            x1_cent - host_len/2 * np.cos(theta1) - host_wid/2 * np.sin(theta1), \
                            x1_cent - host_len/2 * np.cos(theta1) + host_wid/2 * np.sin(theta1), \
                            x1_cent + host_len/2 * np.cos(theta1) + host_wid/2 * np.sin(theta1)])
    host_y_rect = np.array([y1_cent + host_len/2 * np.sin(theta1) - host_wid/2 * np.cos(theta1), \
                            y1_cent + host_len/2 * np.sin(theta1) + host_wid/2 * np.cos(theta1), \
                            y1_cent - host_len/2 * np.sin(theta1) + host_wid/2 * np.cos(theta1), \
                            y1_cent - host_len/2 * np.sin(theta1) - host_wid/2 * np.cos(theta1), \
                            y1_cent + host_len/2 * np.sin(theta1) - host_wid/2 * np.cos(theta1)])


    # compute rectangle corner points of host vehicle
    trailer_x_rect = np.array([x2 + trailer_front_overhang * np.cos(theta2) + trailer_width/2 * np.sin(theta2), \
                                x2 + trailer_front_overhang * np.cos(theta2) - trailer_width/2 * np.sin(theta2), \
                                x2 - trailer_rear_overhang * np.cos(theta2) - trailer_width/2 * np.sin(theta2), \
                                x2 - trailer_rear_overhang * np.cos(theta2) + trailer_width/2 * np.sin(theta2), \
                                x2 + trailer_front_overhang * np.cos(theta2) + trailer_width/2 * np.sin(theta2)])
    trailer_y_rect = np.array([y2 + trailer_front_overhang * np.sin(theta2) - trailer_width/2 * np.cos(theta2), \
                                y2 + trailer_front_overhang * np.sin(theta2) + trailer_width/2 * np.cos(theta2), \
                                y2 - trailer_rear_overhang * np.sin(theta2) + trailer_width/2 * np.cos(theta2), \
                                y2 - trailer_rear_overhang * np.sin(theta2) - trailer_width/2 * np.cos(theta2), \
                                y2 + trailer_front_overhang * np.sin(theta2) - trailer_width/2 * np.cos(theta2)])

    axes.clear()
    
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
    self.ax.set(xlim=(-30, 30), ylim=(-30, 30))

    plt.pause(np.finfo(np.float32).eps)

def close(self):
    ''' '''
    plt.close('all')
    self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
    
# def main():
#     vehicleRender = VehicleRender()
#     vehicleRender.render()
#     vehicleRender.render()
#     vehicleRender.render()

#     plt.show()

# if __name__ == '__main__':
#     main()