Digital Twin of a 3-DOF Industrial Manipulator
Overview
This project implements a Digital Twin of a 3-DOF planar robotic manipulator using Python. The system simulates robot motion, computes forward kinematics, visualizes trajectories, and exports simulation data.
The objective is to demonstrate how a virtual robot model can replicate the behavior of a physical robotic arm in real time.
Features
* 3-DOF planar robotic arm simulation
* Forward kinematics computation
* Jacobian matrix calculation
* Joint position visualization
* Workspace analysis
* Real-time trajectory animation
* Joint angle plotting
* JSON state export
* Configurable robot parameters
Project Structure
* kinematics.py – Forward kinematics, Jacobian, workspace analysis
* digital_twin.py – Simulation engine and state management
* trajectory.py – Visualization and animation
* simulation.json – Configuration and robot parameters

Installation
pip install -r requirements.txt
Run Simulation
python digital_twin.py
Run Visualization
python trajectory.py



Robot Parameters

Parameter Value
DOF	3
Link 1	110 mm
Link 2	90 mm
Link 3	60 mm
Maximum Reach	260 mm


Future Improvements
* Inverse Kinematics
* Collision Detection
* Obstacle Avoidance
* Path Planning
* Real Hardware Integration
