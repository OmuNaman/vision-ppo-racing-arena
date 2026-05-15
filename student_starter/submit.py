"""Evaluate, record, and upload a leaderboard submission."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from eval_maps import EVAL_MAPS
from env import RacingEnv
from model import CNNActorCritic


SUBMIT_MAPS = list(EVAL_MAPS.keys())


def load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def supabase_client():
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in student_starter/.env")
        sys.exit(1)
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: install the pinned requirements first: pip install -r requirements.txt")
        sys.exit(1)
    return create_client(url, key)


def run_episode(policy: CNNActorCritic, env: RacingEnv, device: torch.device, max_steps: int = 1000):
    obs, _info = env.reset()
    total_return = 0.0
    length = 0
    final_info = {}
    while length < max_steps:
        with torch.no_grad():
            dist, _value = policy(torch.as_tensor(obs[None], device=device))
            action = dist.mean.squeeze(0).cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_return += float(reward)
        length += 1
        if terminated or truncated:
            break
    return total_return, length, float(final_info.get("route_completion", 0.0))


def evaluate_variants(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CNNActorCritic.load(checkpoint, device=device)
    policy.eval()

    rows = []
    best = {"return": -float("inf"), "map_name": SUBMIT_MAPS[0]}
    for map_name in SUBMIT_MAPS:
        env = RacingEnv(map_name=map_name, render=False)
        try:
            total_return, length, completion = run_episode(policy, env, device)
        finally:
            env.close()
        rows.append(
            {
                "map_name": map_name,
                "return": float(total_return),
                "episode_length": int(length),
                "route_completion": float(completion),
            }
        )
        if total_return > best["return"]:
            best = {"return": total_return, "map_name": map_name}
        print(
            f"{map_name}: return={total_return:8.2f} "
            f"length={length:4d} route_completion={completion:.1%}"
        )

    return {
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "route_completion": float(np.mean([row["route_completion"] for row in rows])),
        "episodes": rows,
        "best_map": best["map_name"],
    }


def record_topdown_video(checkpoint: str, map_name: str, out_path: str, max_steps: int = 1000) -> str | None:
    try:
        import mediapy
    except ImportError:
        print("mediapy is not installed; skipping video.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CNNActorCritic.load(checkpoint, device=device)
    policy.eval()
    env = RacingEnv(map_name=map_name, render=False)

    class _FakeMainCamera:
        CHASE_TASK_NAME = "_fake_topdown_chase_task"

        def __init__(self, agent):
            self.current_track_agent = agent

        def destroy(self):
            pass

    frames = []
    obs, _info = env.reset()
    md_env = env.unwrapped_metadrive
    agent = getattr(md_env, "agent", None)
    if agent is None and hasattr(md_env, "agents"):
        agent = md_env.agents.get("agent0")
    if agent is not None:
        main_camera = getattr(md_env.engine, "main_camera", None)
        if main_camera is not None and hasattr(main_camera, "current_track_agent"):
            main_camera.current_track_agent = agent
        else:
            md_env.engine.main_camera = _FakeMainCamera(agent)

    try:
        for _step in range(max_steps):
            with torch.no_grad():
                dist, _value = policy(torch.as_tensor(obs[None], device=device))
                action = dist.mean.squeeze(0).cpu().numpy().astype(np.float32)
            obs, _reward, terminated, truncated, _info = env.step(action)
            frame = md_env.render(
                mode="topdown",
                film_size=(3000, 3000),
                screen_size=(900, 900),
                semantic_map=True,
                draw_target_vehicle_trajectory=True,
                draw_contour=True,
            )
            if frame is not None:
                frames.append(np.asarray(frame).swapaxes(0, 1))
            if terminated or truncated:
                break
    finally:
        env.close()

    if not frames:
        print("No top-down frames were captured; skipping video.")
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=30)
    print(f"recorded video -> {out} ({len(frames)} frames)")
    return str(out)


def upload_submission(sb, args: argparse.Namespace, results: dict, video_path: str | None):
    video_url = None
    if video_path:
        video_key = f"{args.uid}/{args.tag}/{int(time.time())}.mp4"
        with open(video_path, "rb") as file:
            sb.storage.from_("videos").upload(video_key, file, {"content-type": "video/mp4"})
        supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
        video_url = f"{supabase_url}/storage/v1/object/public/videos/{video_key}"

    payload = {
        "creator_name": args.name,
        "creator_uid": args.uid,
        "tag": args.tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mean_return": results["mean_return"],
        "mean_episode_length": results["mean_episode_length"],
        "route_completion": results["route_completion"],
        "map_scores": {row["map_name"]: row["route_completion"] for row in results["episodes"]},
        "episode_results": results["episodes"],
        "video_url": video_url,
    }
    response = sb.table("submissions").insert(payload).execute()
    return response.data[0] if response.data else payload


def print_leaderboard(sb=None) -> None:
    print("\nLeaderboard summary")
    if sb is None:
        print("(local only; no upload performed)")
        return
    try:
        rows = (
            sb.table("submissions")
            .select("creator_name,tag,route_completion,mean_return,created_at")
            .order("route_completion", desc=True)
            .limit(10)
            .execute()
            .data
        )
    except Exception as exc:
        print(f"(could not fetch leaderboard: {exc})")
        return
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:2d}. {row.get('creator_name', 'Student'):<20} "
            f"{row.get('tag', ''):<16} completion={float(row.get('route_completion', 0.0)):.1%} "
            f"return={float(row.get('mean_return', 0.0)):8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit your PPO racer to the Supabase leaderboard.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy.pt")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--name", type=str, default="Student")
    parser.add_argument("--uid", type=str, default="student")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Evaluating {checkpoint} on {len(SUBMIT_MAPS)} sky-road variants")
    results = evaluate_variants(str(checkpoint))
    print("\nEvaluation summary")
    print(json.dumps({k: v for k, v in results.items() if k != "episodes"}, indent=2))

    video_path = None
    if not args.no_video:
        video_path = record_topdown_video(str(checkpoint), results["best_map"], f"videos/{args.tag}.mp4")

    if args.no_upload:
        print_leaderboard(None)
        return

    sb = supabase_client()
    submission = upload_submission(sb, args, results, video_path)
    print(f"\nUploaded submission: {submission.get('id', '(no id returned)')}")
    print_leaderboard(sb)


if __name__ == "__main__":
    main()
