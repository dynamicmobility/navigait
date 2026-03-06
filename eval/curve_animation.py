import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SimpleBezier:
    def __init__(self, control_points, num_samples=100):
        assert len(control_points) == 8, "Septic Bézier needs 8 control points"
        self.control_points = control_points
        self.idxs = self.get_idxs(num_samples=num_samples)
        self.points = self.get_bezier_points()

    def bernstein_poly(self, i, n, t):
        from math import comb
        return comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)

    def evaluate_bezier(self, t):
        n = 7
        return sum(
            self.bernstein_poly(i, n, t) * self.control_points[i]
            for i in range(8)
        )
        
    def get_idxs(self, num_samples):
        return np.linspace(0, 1, num_samples)

    def get_bezier_points(self):
        points = [self.evaluate_bezier(t) for t in self.idxs]
        return np.array(points)


class AnimatedBezier:
    def __init__(self, ax: plt.Axes, beziers: list, jt_idx: int, xlim, ylim):
        self.ax = ax
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.patch.set_alpha(0)
        self.curr_point, = ax.plot([], [], color='red', markersize=5, marker='o')
        self.curr_bez,   = ax.plot([], [], color='red')  
        self.old_bez,    = ax.plot([], [], color='black',  alpha=0.8, ls='--')
        self.old_bez2,   = ax.plot([], [], color='black',  alpha=0.3, ls='--')
        self.interp_bez, = ax.plot([], [], color='blue')
        
        self.beziers        = beziers
        self.last_phase     = np.inf
        self.last_bezier    = None
        self.interp_bez_obj = None
        self.interp_idxs    = None
        self.last_idxs      = None
        self.last_vel_idx   = -1
        self.last_vel_idx2  = -1
        self.jt_idx         = jt_idx
        
    def init(self):
        self.curr_point.set_data([], [])
        self.curr_bez.set_data([], [])
        self.old_bez.set_data([], [])
        self.old_bez2.set_data([], [])
        self.interp_bez.set_data([], [])
        return (self.curr_point, 
                self.old_bez, 
                self.old_bez2, 
                self.curr_bez, 
                self.interp_bez)
        
    def update(self, i):
        bezier = SimpleBezier(self.beziers[i][2][self.jt_idx])
        phase = 1 - self.beziers[i][0] / 0.9
        frame_idxs = np.linspace(phase, 1, 100)
        if phase > self.last_phase:
            if self.beziers[i][3] != self.last_vel_idx2:
                self.interp_bez_obj = bezier
                self.interp_idxs = frame_idxs
                self.interp_bez.set_data(self.interp_idxs, self.interp_bez_obj.points)
            self.last_vel_idx2 = self.beziers[i][3]
            self.curr_point.set_data([phase], [bezier.points[0]])
        elif phase == self.last_phase:
            point_phase = self.beziers[i][1] * 0.3 + 0.7
            self.curr_point.set_data([point_phase], [bezier.points[int(self.beziers[i][1] * 100)]])
        self.curr_bez.set_data(frame_idxs, bezier.points)  # wrap in list to make them sequences            
        if self.last_phase > phase:
            if self.beziers[i][3] != self.last_vel_idx and self.last_bezier is not None:
                self.old_bez2.set_data(self.last_idxs, self.last_bezier.points)
            self.old_bez.set_data(frame_idxs, bezier.points)
            self.last_bezier = bezier
            self.last_idxs = frame_idxs
            self.last_vel_idx = self.beziers[i][3]
        self.last_phase = phase
        
        return (self.curr_point,
                self.old_bez,
                self.old_bez2,
                self.curr_bez,
                self.interp_bez)
        
class ManyAnimatedBeziers:
    def __init__(self, axs, beziers):
        self.bezier_objects = []
        YLIMS = {}
        jt_vals = np.array([bez[4] for bez in beziers])
        for idx in range(len(axs)):
            xlim = (-0.1, 1.1)
            ylim = (np.min(jt_vals[:, idx]) - 0.01, 
                    np.max(jt_vals[:, idx]) + 0.01)
            bezier = AnimatedBezier(axs[idx], beziers, idx, xlim, ylim)
            self.bezier_objects.append(bezier)
        
        self.idxs = len(beziers)

    def init(self):
        lines = [bez.init() for bez in self.bezier_objects]
        total_lines = ()
        for line in lines:
            total_lines = total_lines + line
        return total_lines

    def update(self, i):
        print(f'{i/self.idxs:.2%} done\t', end='\r')
        lines = [bez.update(i) for bez in self.bezier_objects]
        total_lines = ()
        for line in lines:
            total_lines = total_lines + line
        return total_lines

def animate_beziers(beziers, vid_length, path):
    # Example curve

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
    axs = axs.flatten()

    many = ManyAnimatedBeziers(axs, beziers)
    ani = animation.FuncAnimation(fig, many.update, frames=len(beziers), init_func=many.init, blit=True, interval=vid_length * 1000 / len(beziers))

    # Save to transparent video
    # ani.save("curve_animation.mov", codec="prores_ks", dpi=300, extra_args=["-pix_fmt", "yuva444p10le"])
    ani.save(path, dpi=200, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuva420p'])