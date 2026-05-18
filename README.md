# Vision-Based PPO Sky-Road Arena

Students train a PPO agent to drive a MetaDrive car from raw 3D RGB camera frames only. The road is a narrow chicane/sky-bridge-style course: obstacles and sparse traffic appear on the route, and leaving the road is treated as falling off the platform.

## Student Goal

Your goal is to train the pixel-only PPO driver to maximize route completion on the default sky-road track. The main lever you are expected to experiment with is the reward design in `reward_config.py`: tune the speed reward, center bonus, brake/idle penalties, crash/fall penalties, and maximum rewarded speed, then retrain and compare route completion.

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

Headless training. This is the mode you should use for real runs because rendering is much slower:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 1000000
```

RTX 4070 / CUDA command:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 500000 --device cuda --horizon 1024 --minibatch 256 --epochs 4 --checkpoint checkpoints\policy.pt
```

Watch the 3D Panda3D window while the car learns:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 1000000 --render
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

- `LANE_WIDTH = 3.4`: road width. Keep it `<= 3.5` for the narrow-road challenge.
- `TRAFFIC_DENSITY = 0.0`, `RANDOM_TRAFFIC = False`, `ACCIDENT_PROB = 0.0`: default is deterministic and learnable. Increase later for extra difficulty.
- `STEERING_SCALE = 0.25` and `THROTTLE_SCALE = 0.85`: internal action scaling before commands reach MetaDrive.
- `MAX_REWARDED_SPEED_KMH = 28.0`: maximum speed rewarded before overspeed penalties begin.
- `BASE_DRIVING_REWARD = 1.60` and `BASE_SPEED_REWARD = 0.15`: MetaDrive's base progress/speed shaping.
- `CENTER_BONUS = 0.25` and `OFF_CENTER_PENALTY = 0.20`: lane-centering reward shaping.
- `SPEED_BONUS = 0.08` and `OVERSPEED_PENALTY = 0.03`: speed shaping.
- `THROTTLE_BONUS = 0.015`: small reward for asking the car to move.
- `BRAKE_BASE_PENALTY = 0.25`, `LOW_SPEED_BRAKE_EXTRA_PENALTY = 0.90`, and `IDLE_BRAKE_EXTRA_PENALTY = 0.90`: brake discouragement.
- `IDLE_STEP_PENALTY = 0.08`, `STALL_AFTER_STEPS = 35`, and `STALL_TERMINAL_REWARD = -80.0`: anti-idle behavior.
- `FINISH_REWARD = 160.0`, `FALL_OFF_ROAD_REWARD = -80.0`, `HIT_OBSTACLE_REWARD = -60.0`, `HIT_TRAFFIC_REWARD = -60.0`, and `HIT_EDGE_REWARD = -80.0`: terminal outcomes.

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

This repo includes a static Supabase-powered leaderboard in `leaderboard/index.html`. It shows the best submission per student, ranks by route completion, plays replay videos, and live-updates when new submissions arrive.

Current deployed leaderboard:

https://leaderboard-ruddy-nine.vercel.app

Instructor setup:

1. Create a new Supabase project.
2. Open Supabase SQL Editor and run `leaderboard/schema.sql`.
3. Confirm Storage has a public bucket named `videos`.
4. Copy your Project URL and publishable key from Supabase Settings -> API.
5. Put the same values in each student's `student_starter/.env`:

```text
SUPABASE_URL=https://nuykrnrijmviisdaxaha.supabase.co
SUPABASE_KEY=sb_publishable_VCZ9nUzqj0qq1GYAZtDAEg_gfhQ5ATW
```

6. Open `leaderboard/index.html` and replace:

```javascript
const SUPABASE_URL = "https://nuykrnrijmviisdaxaha.supabase.co";
const SUPABASE_KEY = "sb_publishable_VCZ9nUzqj0qq1GYAZtDAEg_gfhQ5ATW";
```

7. Deploy the `leaderboard/` folder as a static site.

With Vercel CLI:

```powershell
npm i -g vercel
vercel leaderboard --prod
```

With GitHub Pages:

- push the repo to GitHub
- Settings -> Pages
- deploy from branch
- set folder to `/leaderboard` if your Pages settings allow it, or copy `leaderboard/index.html` into the selected Pages folder

Important: the publishable key is expected to be public. Row-level security policies in `leaderboard/schema.sql` allow public read/insert for the classroom leaderboard. Never publish the Supabase `service_role` or secret key.

## Files

- `env.py`: 3D MetaDrive sky-road wrapper with stacked RGB camera observations shaped `(4, 3, 64, 64)`.
- `eval_maps.py`: The default leaderboard sky-road map config.
- `reward_config.py`: Student-editable reward, penalty, maximum-speed, and action-scaling constants.
- `model.py`: CNN actor-critic with diagonal Normal continuous action policy.
- `train_ppo.py`: PyTorch PPO with GAE, clipping, entropy bonus, TensorBoard logging, gradient clipping, and checkpointing.
- `eval_policy.py`: Deterministic checkpoint evaluation.
- `submit.py`: Evaluation, replay video recording, Supabase upload, and leaderboard summary.
- `../leaderboard/index.html`: Static live leaderboard dashboard.
- `../leaderboard/schema.sql`: Supabase table, policy, and storage setup SQL.
