import numpy as np


class ForwardKinematics:
    """
    2D planar forward kinematics for a 3-DOF serial manipulator.
    Uses the standard DH-convention cumulative angle approach.

    Joint angles are in degrees. Link lengths in millimetres.
    """

    def __init__(self, L1: float, L2: float, L3: float):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def compute(self, theta1_deg: float, theta2_deg: float, theta3_deg: float) -> np.ndarray:
        """
        Compute end-effector (x, y) position given joint angles.

        Args:
            theta1_deg: Base joint angle in degrees.
            theta2_deg: Elbow joint angle in degrees (relative).
            theta3_deg: Wrist joint angle in degrees (relative).

        Returns:
            np.ndarray: [x, y] position of end-effector in mm.
        """
        t1 = np.radians(theta1_deg)
        t2 = np.radians(theta2_deg)
        t3 = np.radians(theta3_deg)

        x = (self.L1 * np.cos(t1)
             + self.L2 * np.cos(t1 + t2)
             + self.L3 * np.cos(t1 + t2 + t3))

        y = (self.L1 * np.sin(t1)
             + self.L2 * np.sin(t1 + t2)
             + self.L3 * np.sin(t1 + t2 + t3))

        return np.array([x, y])

    def joint_positions(self, theta1_deg: float, theta2_deg: float, theta3_deg: float) -> list:
        """
        Returns all joint (x, y) positions including base and end-effector.
        Useful for plotting the full arm geometry.

        Returns:
            List of [x, y] for: base, J1, J2, end-effector.
        """
        t1 = np.radians(theta1_deg)
        t2 = np.radians(theta2_deg)
        t3 = np.radians(theta3_deg)

        p0 = np.array([0.0, 0.0])

        p1 = np.array([
            self.L1 * np.cos(t1),
            self.L1 * np.sin(t1)
        ])

        p2 = p1 + np.array([
            self.L2 * np.cos(t1 + t2),
            self.L2 * np.sin(t1 + t2)
        ])

        p3 = p2 + np.array([
            self.L3 * np.cos(t1 + t2 + t3),
            self.L3 * np.sin(t1 + t2 + t3)
        ])

        return [p0, p1, p2, p3]

    def jacobian(self, theta1_deg: float, theta2_deg: float, theta3_deg: float) -> np.ndarray:
        """
        Compute the 2x3 geometric Jacobian at the given configuration.
        Useful for velocity analysis and singularity detection.

        Returns:
            np.ndarray: 2x3 Jacobian matrix.
        """
        t1 = np.radians(theta1_deg)
        t2 = np.radians(theta2_deg)
        t3 = np.radians(theta3_deg)

        s1 = np.sin(t1)
        s12 = np.sin(t1 + t2)
        s123 = np.sin(t1 + t2 + t3)
        c1 = np.cos(t1)
        c12 = np.cos(t1 + t2)
        c123 = np.cos(t1 + t2 + t3)

        J = np.array([
            [
                -self.L1 * s1 - self.L2 * s12 - self.L3 * s123,
                -self.L2 * s12 - self.L3 * s123,
                -self.L3 * s123
            ],
            [
                self.L1 * c1 + self.L2 * c12 + self.L3 * c123,
                self.L2 * c12 + self.L3 * c123,
                self.L3 * c123
            ]
        ])
        return J

    def workspace_radius(self) -> float:
        """Maximum reach of the end-effector (fully extended)."""
        return self.L1 + self.L2 + self.L3

    def is_reachable(self, x: float, y: float) -> bool:
        """Check if a target point (x, y) lies within the workspace."""
        dist = np.sqrt(x**2 + y**2)
        return dist <= self.workspace_radius()


if __name__ == "__main__":
    fk = ForwardKinematics(L1=110, L2=90, L3=60)

    angles = [(45, -30, 20), (90, -90, 45), (0, 0, 0), (-60, 60, -30)]

    print(f"{'θ1':>6} {'θ2':>6} {'θ3':>6} | {'x (mm)':>8} {'y (mm)':>8} | {'reach':>8}")
    print("-" * 58)
    for t1, t2, t3 in angles:
        ee = fk.compute(t1, t2, t3)
        reach = np.linalg.norm(ee)
        print(f"{t1:>6} {t2:>6} {t3:>6} | {ee[0]:>8.2f} {ee[1]:>8.2f} | {reach:>8.2f}")
