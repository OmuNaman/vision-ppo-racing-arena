# Vision-Based PPO Sky-Road Arena

Students train a PPO agent to drive a MetaDrive car from raw 3D RGB camera frames only. The policy receives four stacked `94x94` RGB camera frames. The default road is a simple sky-bridge curriculum course: a long straight start, then a gentler curve section. Leaving the road is treated as falling off the platform.

## Student Goal

Your goal is to train the pixel-only PPO driver to maximize route completion on the default sky-road track. The main lever you are expected to experiment with is the reward design in `reward_config.py`: tune the progress reward, heading alignment reward, curve-speed reward, center bonus, brake/idle penalties, crash/fall penalties, and maximum rewarded speed, then retrain and compare route completion.

Do not add state vectors, LiDAR, privileged observations, or hand-coded driving rules. The policy must still learn from RGB camera frames only.

The repo includes a live leaderboard dashboard so students can submit checkpoints and compare completion scores.

## Setup In 3 Commands

```powershell
git clone https://github.com/OmuNaman/vision-ppo-racing-arena.git; cd vision-ppo-racing-arena\student_starter
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already cloned the repo, start from the project root:

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

If you have an NVIDIA GPU such as an RTX 4070, install the CUDA PyTorch wheel after installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe check_cuda.py
```

## Train

Headless CPU training. This is the simplest reliable command:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 500000 --device cpu --horizon 512 --minibatch 64 --epochs 2 --lr 0.0001 --target-kl 0.03 --checkpoint checkpoints\policy.pt
```

RTX 4070 / CUDA command:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 500000 --device cuda --horizon 1024 --minibatch 256 --epochs 4 --checkpoint checkpoints\policy.pt
```

Watch the 3D Panda3D window while the car learns:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 500000 --device cpu --horizon 512 --minibatch 64 --epochs 2 --lr 0.0001 --target-kl 0.03 --checkpoint checkpoints\policy.pt --render
```

Checkpoints are saved to `checkpoints/policy.pt` every 50k environment steps and once at the end. TensorBoard logs go to `runs/ppo_pixels`.

To confirm PyTorch sees your RTX 4070:

```powershell
.\.venv\Scripts\python.exe check_cuda.py
```

Note: MetaDrive simulation and camera capture are still the bottleneck, so the GPU will speed up PPO/CNN updates more than the environment itself. Headless training is usually the biggest speed win.

## Reward

All student-editable reward and control values live in `reward_config.py`. Change that file when you want to tune the behavior. After editing it, restart training so the new values are loaded. The score to optimize is route completion, reported during training/evaluation as `completion` or `route_completion`.

Important values include:

- `LANE_WIDTH = 4.2`: road width. Lower it later if you want to make the challenge harder.
- `TRAFFIC_DENSITY = 0.0`, `RANDOM_TRAFFIC = False`, `ACCIDENT_PROB = 0.0`: default is deterministic and learnable. Increase later for extra difficulty.
- `STEERING_SCALE = 0.22` and `THROTTLE_SCALE = 0.75`: internal action scaling before commands reach MetaDrive.
- `PROGRESS_REWARD_SCALE = 350.0`: main reward for increasing route completion.
- `BACKWARD_PROGRESS_PENALTY_SCALE = 80.0`: penalty scale if completion goes backward.
- `MAX_REWARDED_SPEED_KMH = 28.0`: maximum speed rewarded before overspeed penalties begin.
- `BASE_DRIVING_REWARD = 0.0` and `BASE_SPEED_REWARD = 0.0`: MetaDrive base reward is disabled so the signal stays simple.
- `CENTER_BONUS = 0.04` and `OFF_CENTER_PENALTY = 0.08`: small lane-centering shaping.
- `HEADING_ALIGNMENT_BONUS = 0.16` and `HEADING_ERROR_PENALTY = 0.12`: reward-only lane-tangent guidance so the car learns to face the road before it falls.
- `CURVE_LOOKAHEAD_METERS = 18.0`, `CURVE_CENTER_BONUS = 0.08`, `CURVE_SPEED_LIMIT_KMH = 20.0`, and `CURVE_OVERSPEED_PENALTY = 0.08`: curve-aware shaping.
- `SPEED_BONUS = 0.02` and `OVERSPEED_PENALTY = 0.03`: light speed shaping.
- `THROTTLE_BONUS = 0.005`: tiny reward for asking the car to move.
- `STEERING_BASE_PENALTY = 0.006`, `STEERING_SPEED_PENALTY = 0.010`, and `CURVE_STEERING_RELIEF = 0.85`: steering is only lightly discouraged, and even less on curves.
- `BRAKE_BASE_PENALTY = 0.05`, `LOW_SPEED_BRAKE_EXTRA_PENALTY = 0.20`, and `IDLE_BRAKE_EXTRA_PENALTY = 0.20`: brake discouragement.
- `IDLE_STEP_PENALTY = 0.03`, `STALL_AFTER_STEPS = 35`, and `STALL_TERMINAL_REWARD = -20.0`: anti-idle behavior.
- `FINISH_REWARD = 100.0`, `FALL_OFF_ROAD_REWARD = -20.0`, `HIT_OBSTACLE_REWARD = -20.0`, `HIT_TRAFFIC_REWARD = -20.0`, and `HIT_EDGE_REWARD = -20.0`: terminal outcomes.

The reward is intentionally simple:

- Increasing route completion is the main reward.
- Aligning the vehicle heading with the current lane tangent gets a reward.
- Curves are detected with a short lane lookahead; on curves, center-keeping matters more and steering penalties are softened.
- Overspeeding above `MAX_REWARDED_SPEED_KMH` gets a penalty.
- Overspeeding through curves gets an extra penalty.
- Staying near the lane center gets a small bonus.
- Positive throttle gets a small bonus.
- Braking is penalized, especially at low speed or when already nearly idle.
- Excessive steering is lightly penalized, especially while fast or near the road edge.
- Steering and positive throttle commands are scaled down internally so early PPO exploration is less twitchy.
- Crawling/idling after the first few steps gets a small recurring penalty.
- Sustained low-speed stalling ends the episode with a penalty.
- Reaching the finish gives `+100`.
- Falling off the road gives `-20`.
- Hitting an obstacle or traffic vehicle gives `-20`.

## Smoke-Tested Numbers

These are not final-performance claims; they are sanity checks that the environment now gives PPO a usable learning signal.

Current fixed-action reward sanity check:

- straight throttle: return `242.30`, length `199`, route completion `72.8%`
- idle: return `-10.21`, length `55`, route completion `1.4%`

PPO smoke tests on CPU after the `94x94` vision and curve-reward update:

- Environment reset/model-forward check passed with observation shape `(4, 3, 94, 94)`.
- Fixed-action straight throttle now passes the old curve wall and reaches `72.8%` before falling.
- 2,048 headless training steps completed and saved `checkpoints/vision_curve_smoke.pt`.
- The 2,048-step deterministic smoke checkpoint reached `9.5%` mean route completion over 3 episodes. This is only a wiring test, not a performance target; train much longer for a real checkpoint.

## Evaluate

Evaluate only after `checkpoints/policy.pt` exists. If you see `FileNotFoundError: checkpoints\policy.pt`, train first.

```powershell
.\.venv\Scripts\python.exe eval_policy.py --checkpoint checkpoints/policy.pt
.\.venv\Scripts\python.exe eval_policy.py --checkpoint checkpoints/policy.pt --render
```

Evaluation runs 20 episodes and reports mean return, mean episode length, and route completion percentage.

## Submit

Create `.env` inside `student_starter`. For this class leaderboard you can copy the provided `.env.example`:

```text
SUPABASE_URL=https://nuykrnrijmviisdaxaha.supabase.co
SUPABASE_KEY=sb_publishable_VCZ9nUzqj0qq1GYAZtDAEg_gfhQ5ATW
```

Then submit:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag first-run --name "Your Name" --uid your-id
```

`submit.py` evaluates on the default sky-road map, records an MP4 replay for the leaderboard, uploads the result to the Supabase `submissions` table, then prints a leaderboard summary.

The default replay is a top-down MetaDrive video because it is reliable in headless mode. To record the actual RGB camera frames seen by PPO instead:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag camera-view --video-mode camera
```

For a local dry run without upload:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag dry-run --no-upload
```

## Live Leaderboard

Submissions appear on the official class leaderboard. Students only need to train, submit, and view their score here:

https://leaderboard-ruddy-nine.vercel.app

Use these values in `student_starter/.env` so `submit.py` can upload your evaluated score and replay video:

```text
SUPABASE_URL=https://nuykrnrijmviisdaxaha.supabase.co
SUPABASE_KEY=sb_publishable_VCZ9nUzqj0qq1GYAZtDAEg_gfhQ5ATW
```

## Files

- `env.py`: 3D MetaDrive sky-road wrapper with stacked RGB camera observations shaped `(4, 3, 94, 94)`.
- `eval_maps.py`: The default leaderboard sky-road map config.
- `reward_config.py`: Student-editable reward, penalty, maximum-speed, and action-scaling constants.
- `model.py`: CNN actor-critic with diagonal Normal continuous action policy.
- `train_ppo.py`: PyTorch PPO with GAE, clipping, entropy bonus, TensorBoard logging, gradient clipping, and checkpointing.
- `eval_policy.py`: Deterministic checkpoint evaluation.
- `submit.py`: Evaluation, replay video recording, Supabase upload, and leaderboard summary.
