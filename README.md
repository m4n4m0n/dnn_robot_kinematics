# dnn_robot_kinematics
This project implements a training pipeline for deep neural networks (DNN) to predict either forward or inverse kinematics of a robot arm. 


## Project Overview

- **Forward Kinematics (FK):**
    - Input: Joint angles `q_solution`
    - Output: Cartesian positions + orientation (translation + quaternion)

- **Inverse Kinematics (IK):**
    - Input: Cartesian positions + orientation (translation + quaternion)
    - Output: Joint angles `q_solution`

The data is stored in `data.pkl` and can also be loaded from CSV using `load_csv()`.

---

## How to Run

### 1. Install Dependencies

Create a virtual environment and install packages:

```bash
pip install numpy matplotlib torch scikit-learn
