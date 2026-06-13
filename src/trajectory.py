import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinematics import ForwardKinematics


class TrajectoryPlanner:
    """
    Generates joint-space trajectories and plots them.
    Supports sinusoidal auto-motion and linear interpolation
    between waypoints.
    """

    def __init__(self, L1=110, L2=90, L3=60):
        self.fk = ForwardKinematics(L1, L2, L3)
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def sinusoidal_trajectory(self, duration=10.0, dt=0.05):
        """Generate a sinusoidal joint-space trajectory."""
        t = np.arange(0, duration, dt)
        t1 = 45 * np.sin(t * 0.6) + 30 * np.sin(t * 0.3)
        t2 = -40 * np.sin(t * 0.8 + 1) + 20 * np.cos(t * 0.4)
        t3 = 30 * np.sin(t * 1.1 + 0.5)
        return t, t1, t2, t3

    def linear_interpolate(self, waypoints, steps_per_segment=50):
        """
        Linearly interpolate between a list of joint-angle waypoints.

        Args:
            waypoints: list of (t1, t2, t3) in degrees
            steps_per_segment: resolution per segment

        Returns:
            np.ndarray of shape (N, 3)
        """
        trajectory = []
        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i])
            end = np.array(waypoints[i + 1])
            for s in np.linspace(0, 1, steps_per_segment, endpoint=False):
                trajectory.append(start + s * (end - start))
        trajectory.append(np.array(waypoints[-1]))
        return np.array(trajectory)

    def plot_static(self, theta1, theta2, theta3, title="3-DOF Arm Configuration"):
        """Plot a static arm pose."""
        pts = self.fk.joint_positions(theta1, theta2, theta3)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        colors = ["#7F77DD", "#1D9E75", "#EF9F27"]
        labels = ["Link 1 (Base)", "Link 2 (Elbow)", "Link 3 (Wrist)"]
        for i in range(3):
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color=colors[i], linewidth=4, solid_capstyle='round', label=labels[i])

        joint_colors = ["#7F77DD", "#1D9E75", "#EF9F27"]
        for i, (x, y) in enumerate(zip(xs[:-1], ys[:-1])):
            ax.scatter(x, y, color=joint_colors[i], s=80, zorder=5)

        ax.scatter(xs[-1], ys[-1], color="#E24B4A", s=120, marker="^", zorder=6, label="End-effector")

        R = self.L1 + self.L2 + self.L3
        theta_ws = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(theta_ws), R * np.sin(theta_ws),
                color="#7F77DD", linewidth=0.5, linestyle="--", alpha=0.4, label="Workspace")

        ax.axhline(0, color="#30363d", linewidth=0.5)
        ax.axvline(0, color="#30363d", linewidth=0.5)
        ax.set_xlim(-320, 320)
        ax.set_ylim(-320, 320)
        ax.set_aspect("equal")
        ax.set_title(title, color="#e6edf3", fontsize=13)
        ax.set_xlabel("X (mm)", color="#8b949e")
        ax.set_ylabel("Y (mm)", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        ee = self.fk.compute(theta1, theta2, theta3)
        ax.set_title(
            f"{title}\nEE = ({ee[0]:.1f}, {ee[1]:.1f}) mm | θ=({theta1}°, {theta2}°, {theta3}°)",
            color="#e6edf3", fontsize=11
        )
        plt.tight_layout()
        plt.savefig("arm_pose.png", dpi=150, bbox_inches="tight")
        plt.show()

    def plot_joint_angles(self, duration=10.0):
        """Plot joint angles over time (like MATLAB's plot)."""
        t, t1, t2, t3 = self.sinusoidal_trajectory(duration)

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        fig.patch.set_facecolor("#0d1117")

        data = [(t1, "#7F77DD", "θ₁ — Base (°)"),
                (t2, "#1D9E75", "θ₂ — Elbow (°)"),
                (t3, "#EF9F27", "θ₃ — Wrist (°)")]

        for ax, (vals, col, ylabel) in zip(axes, data):
            ax.set_facecolor("#161b22")
            ax.plot(t, vals, color=col, linewidth=1.5)
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
            ax.tick_params(colors="#8b949e")
            ax.grid(True, color="#30363d", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        axes[-1].set_xlabel("Time (s)", color="#8b949e")
        fig.suptitle("Joint Angles over Time — Digital Twin", color="#e6edf3", fontsize=12)
        plt.tight_layout()
        plt.savefig("joint_angles.png", dpi=150, bbox_inches="tight")
        plt.show()

    def animate(self, duration=8.0, interval=50):
        """Live animated simulation of the 3-DOF arm."""
        t, t1_arr, t2_arr, t3_arr = self.sinusoidal_trajectory(duration, dt=interval/1000)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        R = self.L1 + self.L2 + self.L3
        theta_ws = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(theta_ws), R * np.sin(theta_ws),
                color="#7F77DD", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.axhline(0, color="#30363d", linewidth=0.5)
        ax.axvline(0, color="#30363d", linewidth=0.5)
        ax.set_xlim(-320, 320)
        ax.set_ylim(-320, 320)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)", color="#8b949e")
        ax.set_ylabel("Y (mm)", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        colors = ["#7F77DD", "#1D9E75", "#EF9F27"]
        lines = [ax.plot([], [], color=c, linewidth=4, solid_capstyle='round')[0] for c in colors]
        joints = [ax.plot([], [], 'o', color=c, markersize=8)[0] for c in colors]
        ee_dot, = ax.plot([], [], '^', color="#E24B4A", markersize=10)
        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, color="#8b949e", fontsize=9)
        ee_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color="#E24B4A", fontsize=9)

        trace_x, trace_y = [], []
        trace, = ax.plot([], [], color="#E24B4A", linewidth=0.8, alpha=0.4)

        def init():
            for l in lines + joints:
                l.set_data([], [])
            ee_dot.set_data([], [])
            trace.set_data([], [])
            return lines + joints + [ee_dot, trace, time_text, ee_text]

        def update(frame):
            pts = self.fk.joint_positions(t1_arr[frame], t2_arr[frame], t3_arr[frame])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            for i, line in enumerate(lines):
                line.set_data([xs[i], xs[i+1]], [ys[i], ys[i+1]])
            for i, j in enumerate(joints):
                j.set_data([xs[i]], [ys[i]])
            ee_dot.set_data([xs[-1]], [ys[-1]])
            trace_x.append(xs[-1])
            trace_y.append(ys[-1])
            if len(trace_x) > 200:
                trace_x.pop(0)
                trace_y.pop(0)
            trace.set_data(trace_x, trace_y)
            time_text.set_text(f"t = {t[frame]:.2f}s")
            ee_text.set_text(f"EE ({xs[-1]:.0f}, {ys[-1]:.0f}) mm")
            return lines + joints + [ee_dot, trace, time_text, ee_text]

        ani = animation.FuncAnimation(fig, update, frames=len(t),
                                      init_func=init, interval=interval, blit=True)
        fig.suptitle("3-DOF Digital Twin — Live Simulation", color="#e6edf3", fontsize=12)
        plt.tight_layout()
        plt.show()
        return ani


if __name__ == "__main__":
    planner = TrajectoryPlanner(L1=110, L2=90, L3=60)
    planner.plot_static(45, -30, 20)
    planner.plot_joint_angles(duration=10)
    planner.animate(duration=8)
