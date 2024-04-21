import numpy as np

# Environment constants
BOUNDARY = np.array([[-0.36, 0.42], [-0.76, -0.24], [0.005, 0.2]])
PIXEL_SIZE = 0.002

# ======== search constants ========
# Unify the translation and rotation weights
TRANS_WEIGHT = 1
ROT_WEIGHT = 0.6

# define cost of robot 
PICK_PLACE_BASE_COST = 0.3
PICK_PLACE_DIST_SCALE = 0.06
PICK_DRAG_BASE_COST = 0.2
PICK_DRAG_DIST_SCALE = 0.1

# MCTS constants two states are considered the same if they are within the threshold
IS_CLOSE_TRANS_THRESHOLD = 0.015
IS_CLOSE_ROT_THRESHOLD = np.deg2rad(10)

# MCTS Exploration constant
C_PUCT = 1.5
Q_LIST_SIZE = 100

# depth = 3
# C_PUCT = 0.5
# Q_LIST_SIZE = 50

# time=10
# C_PUCT = 1.0
# Q_LIST_SIZE = 30

# time=20
# C_PUCT = 1.0
# Q_LIST_SIZE = 20

# # no base_reward
# C_PUCT = 2.0
# Q_LIST_SIZE = 100

# time=60
# C_PUCT = 1.8
# Q_LIST_SIZE = 100
# ======================================
