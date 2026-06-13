import numpy as np
import time
import json
from kinematics import ForwardKinematics

class DigitalTwin:
    def __init__(self):
        self.L1 = 110  # mm
        self.L2 = 90
        self.L3 = 60
        self.fk = ForwardKinematics(self.L1, self.L2, self.L3)
        self.joint_angles = np.array([0.0, 0.0, 0.0])  # degrees
        self.ee_position = np.array([0.0, 0.0])
        self.sim_time = 0.0
        self.running = False
        self.log = []

    def log_msg(self, level, msg):
        ts = time.strftime("[%H:%M:%S]")
        entry = f"{ts} [{level}] {msg}"
        self.log.append(entry)
        print(entry)

    def start(self):
        self.running = True
        self.log_msg("INFO", "Digital Twin simulation started. DOF=3")
        self.log_msg("INFO", f"Link lengths: L1={self.L1} L2={self.L2} L3={self.L3} mm")

    def stop(self):
        self.running = False
        self.log_msg("INFO", "Simulation stopped.")

    def reset(self):
        self.joint_angles = np.array([0.0, 0.0, 0.0])
        self.ee_position = np.array([0.0, 0.0])
        self.sim_time = 0.0
        self.log_msg("OK", "Reset complete — all joints at zero.")

    def set_joints(self, t1, t2, t3):
        self.joint_angles = np.array([t1, t2, t3])
        self.ee_position = self.fk.compute(t1, t2, t3)
        self.log_msg("INFO", f"Joints set: θ1={t1:.1f}° θ2={t2:.1f}° θ3={t3:.1f}°")
        self.log_msg("INFO", f"EE position: x={self.ee_position[0]:.1f} mm, y={self.ee_position[1]:.1f} mm")

    def auto_trajectory(self, duration=10.0, dt=0.05):
        self.start()
        t = 0.0
        while t < duration:
            if not self.running:
                break
            t1 = 45 * np.sin(t * 0.6) + 30 * np.sin(t * 0.3)
            t2 = -40 * np.sin(t * 0.8 + 1) + 20 * np.cos(t * 0.4)
            t3 = 30 * np.sin(t * 1.1 + 0.5)
            self.set_joints(t1, t2, t3)
            self.sim_time = t
            time.sleep(dt)
            t += dt
        self.stop()

    def get_state(self):
        return {
            "sim_time": round(self.sim_time, 3),
            "joint_angles_deg": self.joint_angles.tolist(),
            "joint_angles_rad": np.radians(self.joint_angles).tolist(),
            "ee_x_mm": round(float(self.ee_position[0]), 2),
            "ee_y_mm": round(float(self.ee_position[1]), 2),
            "reach_mm": round(float(np.linalg.norm(self.ee_position)), 2),
        }

    def export_state(self, filepath="simulation.json"):
        with open(filepath, "w") as f:
            json.dump(self.get_state(), f, indent=2)
        self.log_msg("OK", f"State exported to {filepath}")


if __name__ == "__main__":
    twin = DigitalTwin()
    twin.set_joints(45, -30, 20)
    print(json.dumps(twin.get_state(), indent=2))
    twin.auto_trajectory(duration=5.0)
