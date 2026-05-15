# Vision-Based PPO Racing Arena

Students train a PPO agent to drive a MetaDrive car from raw RGB camera frames only. The policy never receives lidar, route vectors, vehicle state, or privileged simulator observations.

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

Watch the Panda3D 3D window while the car learns:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 1000000 --render
```

Checkpoints are saved to `checkpoints/policy.pt` every 50k environment steps and once at the end. TensorBoard logs go to `runs/ppo_pixels`.

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

`submit.py` evaluates on 5 winding track variants, records a top-down MP4 with MetaDrive's topdown renderer and `mediapy`, uploads the result to the Supabase `submissions` table, then prints a leaderboard summary.

For a local dry run without upload:

```powershell
.\.venv\Scripts\python.exe submit.py --checkpoint checkpoints/policy.pt --tag dry-run --no-upload
```

## Files

- `env.py`: MetaDrive wrapper with 64x64 RGB camera input, narrow S-curve/chicane maps, deque frame stacking to `(4, 3, 64, 64)`, and custom reward.
- `model.py`: CNN actor-critic with diagonal Normal continuous action policy.
- `train_ppo.py`: PyTorch PPO with GAE, clipping, entropy bonus, TensorBoard logging, gradient clipping, and checkpointing.
- `eval_policy.py`: Deterministic checkpoint evaluation.
- `submit.py`: Multi-map evaluation, top-down video recording, Supabase upload, and leaderboard summary.
