"""Student-editable reward and control settings for the sky-road arena.

Tune this file first when you want different learning behavior. The PPO policy
still receives only RGB camera pixels; these values only shape the reward and
the low-level control limits inside the environment.
"""

# ----------------------------- Action shaping -----------------------------

# PPO outputs steering in [-1, 1]. MetaDrive receives this scaled value. Keep
# this conservative for early learning; random camera policies otherwise spin
# off the sky road before seeing useful reward.
STEERING_SCALE = 0.16

# PPO outputs throttle/brake in [-1, 1]. Positive values are throttle and are
# multiplied by this before MetaDrive sees them. Negative values remain brake.
THROTTLE_SCALE = 0.75


# ---------------------------- Course difficulty ---------------------------

# Wider is easier, narrower is harder. Keep <= 3.5 for the sky-road challenge.
LANE_WIDTH = 3.5

# Default to a deterministic, learnable road. Increase these only after the
# agent can drive the base track reliably.
TRAFFIC_DENSITY = 0.0
RANDOM_TRAFFIC = False
ACCIDENT_PROB = 0.0
STATIC_TRAFFIC_OBJECTS = False


# -------------------------- MetaDrive base reward --------------------------

# These go directly into MetaDrive's own reward config.
BASE_DRIVING_REWARD = 0.0
BASE_SPEED_REWARD = 0.0
USE_LATERAL_REWARD = True

BASE_SUCCESS_REWARD = 0.0
BASE_OUT_OF_ROAD_PENALTY = 0.0
BASE_CRASH_VEHICLE_PENALTY = 0.0
BASE_CRASH_OBJECT_PENALTY = 0.0
BASE_CRASH_SIDEWALK_PENALTY = 0.0


# ---------------------------- Sky-road shaping -----------------------------

# Main curriculum reward. Route completion is in [0, 1], so this turns small
# forward progress into a clear dense learning signal.
PROGRESS_REWARD_SCALE = 350.0
BACKWARD_PROGRESS_PENALTY_SCALE = 80.0

# Speed target used by the custom reward. Above this, speed gets penalized.
MAX_REWARDED_SPEED_KMH = 28.0

# Bonus for staying close to lane center.
CENTER_BONUS = 0.04

# Quadratic penalty for being away from lane center.
OFF_CENTER_PENALTY = 0.08

# Small dense bonus for moving, capped at MAX_REWARDED_SPEED_KMH.
SPEED_BONUS = 0.02

# Penalty for exceeding MAX_REWARDED_SPEED_KMH.
OVERSPEED_PENALTY = 0.03
OVERSPEED_PENALTY_STEP_KMH = 5.0

# Extra reward for asking for positive throttle. This prevents the policy from
# discovering lazy idle/brake strategies too easily.
THROTTLE_BONUS = 0.005
THROTTLE_BONUS_WHEN_OVERSPEED = 0.25

# Steering penalties. The final steering penalty is:
# base + speed-sensitive + edge-sensitive.
STEERING_BASE_PENALTY = 0.025
STEERING_SPEED_PENALTY = 0.035
STEERING_SPEED_REFERENCE_KMH = 25.0
STEERING_EDGE_PENALTY = 0.08

# Brake penalties. Brake is useful sometimes, but unnecessary braking was a
# common failure mode, so low-speed braking is punished harder.
BRAKE_BASE_PENALTY = 0.05
LOW_SPEED_BRAKE_THRESHOLD_KMH = 12.0
LOW_SPEED_BRAKE_EXTRA_PENALTY = 0.20
IDLE_BRAKE_THRESHOLD_KMH = 3.0
IDLE_BRAKE_EXTRA_PENALTY = 0.20

# Idle/stall behavior.
IDLE_AFTER_STEPS = 20
IDLE_SPEED_THRESHOLD_KMH = 2.0
IDLE_STEP_PENALTY = 0.03

STALL_SPEED_THRESHOLD_KMH = 2.5
STALL_THROTTLE_THRESHOLD = 0.2
STALL_AFTER_STEPS = 35
STALL_TERMINAL_REWARD = -20.0


# ---------------------------- Terminal rewards ----------------------------

FINISH_REWARD = 100.0
FALL_OFF_ROAD_REWARD = -20.0
HIT_OBSTACLE_REWARD = -20.0
HIT_TRAFFIC_REWARD = -20.0
HIT_EDGE_REWARD = -20.0
