# Vision-Based PPO Sky-Road Arena

Students train a PPO agent to drive a MetaDrive car from raw 3D RGB camera frames only. The road is a narrow chicane/sky-bridge-style course: obstacles and sparse traffic appear on the route, and leaving the road is treated as falling off the platform.

## Student Goal

Your goal is to train the pixel-only PPO driver to maximize route completion on the default sky-road track. The main lever you are expected to experiment with is the reward design in `reward_config.py`: tune the speed reward, center bonus, brake/idle penalties, crash/fall penalties, and maximum rewarded speed, then retrain and compare route completion.

Do not add state vectors, LiDAR, privileged observations, or hand-coded driving rules. The policy must still learn from RGB camera frames only.

Upcoming update: a live leaderboard will be added soon so students can submit checkpoints and compare completion scores.

## Setup In 3 Commands

```powershell
cd student_starter
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, replace the second command with `python3.11 -m venv .venv && source .venv/bin/activate`.

If PowerShell shows `(.venv)` but `python -m pip` still points at Python 3.13, use the explicit venv executable:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

Headless training:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 1000000
```

Watch the 3D Panda3D window while the car learns:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 1000000 --render
```

Checkpoints are saved to `checkpoints/policy.pt` every 50k environment steps and once at the end. TensorBoard logs go to `runs/ppo_pixels`.

## Reward

All student-editable reward and control values live in `reward_config.py`. Change that file when you want to tune the behavior. After editing it, restart training so the new values are loaded. The score to optimize is route completion, reported during training/evaluation as `completion` or `route_completion`.

Important values include:

- `MAX_REWARDED_SPEED_KMH`: maximum speed rewarded before overspeed penalties begin.
- `STEERING_SCALE` and `THROTTLE_SCALE`: internal action scaling before commands reach MetaDrive.
- `BASE_DRIVING_REWARD` and `BASE_SPEED_REWARD`: MetaDrive's base progress/speed shaping.
- `CENTER_BONUS` and `OFF_CENTER_PENALTY`: lane-centering reward shaping.
- `SPEED_BONUS` and `OVERSPEED_PENALTY`: speed shaping.
- `THROTTLE_BONUS`: reward for asking the car to move.
- `BRAKE_BASE_PENALTY`, `LOW_SPEED_BRAKE_EXTRA_PENALTY`, and `IDLE_BRAKE_EXTRA_PENALTY`: brake discouragement.
- `IDLE_STEP_PENALTY`, `STALL_AFTER_STEPS`, and `STALL_TERMINAL_REWARD`: anti-idle behavior.
- `FINISH_REWARD`, `FALL_OFF_ROAD_REWARD`, `HIT_OBSTACLE_REWARD`, `HIT_TRAFFIC_REWARD`, and `HIT_EDGE_REWARD`: terminal outcomes.

The reward uses MetaDrive's forward-driving reward as the base, then adds sky-road shaping:

- Forward progress and speed are rewarded.
- Overspeeding above `MAX_REWARDED_SPEED_KMH` gets a penalty.
- Staying near the lane center gets a small bonus.
- Positive throttle gets a small bonus.
- Braking is penalized, especially at low speed or when already nearly idle.
- Excessive steering is penalized, especially while fast or near the road edge.
- Steering and positive throttle commands are scaled down internally so early PPO exploration is less twitchy.
- Crawling/idling after the first few steps gets a recurring penalty.
- Sustained low-speed stalling ends the episode with a penalty.
- Reaching the finish gives `+100`.
- Falling off the road gives `-60`.
- Hitting an obstacle or traffic vehicle gives `-35`.

## Evaluate

```powershell
.\.venv\Scripts\python.exe eval_policy.py --checkpoint checkpoints/policy.pt
.\.venv\Scripts\python.exe eval_policy.py --checkpoint checkpoints/policy.pt --render
```

Evaluation runs 20 episodes and reports mean return, mean episode length, and route completion percentage.

## Submit

Create `student_starter/.env`:

```text
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-service-or-anon-key
```

Then submit:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag first-run --name "Your Name" --uid your-id
```

`submit.py` evaluates on the default sky-road map, records a top-down MP4 for the leaderboard, uploads the result to the Supabase `submissions` table, then prints a leaderboard summary.

For a local dry run without upload:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag dry-run --no-upload
```

## Files

- `env.py`: 3D MetaDrive sky-road wrapper with stacked RGB camera observations shaped `(4, 3, 64, 64)`.
- `eval_maps.py`: The default leaderboard sky-road map config.
- `reward_config.py`: Student-editable reward, penalty, maximum-speed, and action-scaling constants.
- `model.py`: CNN actor-critic with diagonal Normal continuous action policy.
- `train_ppo.py`: PyTorch PPO with GAE, clipping, entropy bonus, TensorBoard logging, gradient clipping, and checkpointing.
- `eval_policy.py`: Deterministic checkpoint evaluation.
- `submit.py`: Evaluation, top-down video recording, Supabase upload, and leaderboard summary.
