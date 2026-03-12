from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import loguru
import mujoco
import numpy as np
import torch
import tyro

from spider.config import Config, process_config
from spider.io import load_data
from spider.simulators.mjwp import (
    get_qpos,
    get_reward,
    get_terminal_reward,
    get_terminate,
    load_state,
    save_state,
    step_env,
)
from spider.viewers import render_image
from spider.rl.mjwp_rsl_env import MJWPRLEnvCfg, MJWPRslResidualEnv

@dataclass
class RLTrainArgs:
    # Dataset / task
    dataset_dir: str = "example_datasets"
    dataset_name: str = "gigahand"
    robot_type: str = "xhand"
    embodiment_type: str = "right"
    task: str = "mustard_901_1"
    data_id: int = 0

    # Timing / planning (align with MPC)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.4
    horizon: float = 1.6
    knot_dt: float = 0.4

    # Reward scales (follow run_aria.sh)
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0
    contact_rew_scale: float = 0.0

    # Termination thresholds (MPC logic)
    #TODO: this might be too strong?
    object_pos_threshold: float = 0.05
    object_rot_threshold: float = 1.0

    # Domain randomization (match mjwp run_mjwp.py logic / parameterization)
    num_dr: int = 1
    pair_margin_range_min: float = -0.005
    pair_margin_range_max: float = 0.005
    xy_offset_range_min: float = -0.005
    xy_offset_range_max: float = 0.005
    object_mass_scale_range_min: float = 1.0
    object_mass_scale_range_max: float = 1.0

    # Residual action scale base (use MPC noise scales)
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03

    # RL
    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 8912
    future_ctrl_steps: int = 0  # kept for compatibility; minimal obs ignores future steps
    max_episode_ctrl_steps: int = 64
    num_steps_per_env: int = 24
    learning_iterations: int = 1000
    ckpt_interval: int = 50
    viz_interval: int = 10
    rollout_video_ctrl_steps: int = 32
    # Optional lightweight W&B media upload (curve + rollout videos only)
    wandb_media_only: bool = False
    wandb_project: str = "spider-rl"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    log_dir: str = "outputs/rl_rsl"
    experiment_name: str = task

def _make_spider_config(args: RLTrainArgs) -> Config:
    cfg = Config(
        simulator="mjwp",
        dataset_dir=os.path.abspath(args.dataset_dir),
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        embodiment_type=args.embodiment_type,
        task=args.task,
        data_id=args.data_id,
        seed=args.seed,
        device=args.device,
        sim_dt=args.sim_dt,
        ctrl_dt=args.ctrl_dt,
        horizon=args.horizon,
        knot_dt=args.knot_dt,
        num_samples=args.num_envs,  # reuse MJWP vector worlds as RL env count
        # reward
        pos_rew_scale=args.pos_rew_scale,
        rot_rew_scale=args.rot_rew_scale,
        joint_rew_scale=args.joint_rew_scale,
        base_pos_rew_scale=args.base_pos_rew_scale,
        base_rot_rew_scale=args.base_rot_rew_scale,
        vel_rew_scale=args.vel_rew_scale,
        contact_rew_scale=args.contact_rew_scale,
        # terminate thresholds
        object_pos_threshold=args.object_pos_threshold,
        object_rot_threshold=args.object_rot_threshold,
        # domain randomization (same parameterization as mjwp)
        num_dr=args.num_dr,
        pair_margin_range=(args.pair_margin_range_min, args.pair_margin_range_max),
        xy_offset_range=(args.xy_offset_range_min, args.xy_offset_range_max),
        object_mass_scale_range=(
            args.object_mass_scale_range_min,
            args.object_mass_scale_range_max,
        ),
        # noise scales for residual action scaling
        first_ctrl_noise_scale=args.first_ctrl_noise_scale,
        last_ctrl_noise_scale=args.last_ctrl_noise_scale,
        final_noise_scale=args.final_noise_scale,
        joint_noise_scale=args.joint_noise_scale,
        pos_noise_scale=args.pos_noise_scale,
        rot_noise_scale=args.rot_noise_scale,
        # disable mpc-only outputs / viewer
        show_viewer=False,
        save_video=False,
        save_info=False,
        save_metrics=False,
        save_rerun=False,
        use_torch_compile=False,
    )
    return process_config(cfg)


def _make_rsl_train_cfg(args: RLTrainArgs) -> dict:
    # rsl_rl (newer API) expects top-level actor/critic/algorithm/obs_groups.
    return {
        "seed": int(args.seed),
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.ckpt_interval),
        "multi_gpu": None,
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo:PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "obs_normalization": True,
            #for the new version of rsl_rl libs
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "obs_normalization": True,
            #TODO: check the version of rsl_rl
            # "stochastic": False,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
    }


def _install_training_viz_hook(
    runner,
    env: MJWPRslResidualEnv,
    log_dir: Path,
    viz_interval: int,
    rollout_video_ctrl_steps: int,
    wandb_run=None,
):
    """Hook rsl_rl logger.log to persist reward snapshots and optional plots."""
    if viz_interval <= 0:
        return []

    viz_dir = log_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    orig_log = runner.logger.log
    warned_video_backend = {"done": False}
    latest_rollout_video_path: Path | None = None

    def _wandb_upload(it: int, upload_video: bool):
        if wandb_run is None:
            return
        media = {}
        curve_png = viz_dir / "reward_curve_latest.png"
        if curve_png.exists():
            try:
                import wandb  # type: ignore

                media["reward_curve"] = wandb.Image(str(curve_png))
            except Exception as e:
                loguru.logger.warning("wandb curve upload failed at iter {}: {}", it, e)
        if upload_video and latest_rollout_video_path is not None and latest_rollout_video_path.exists():
            try:
                import wandb  # type: ignore

                media["rollout_video"] = wandb.Video(str(latest_rollout_video_path), fps=max(1, int(round(1.0 / env.cfg.sim_dt))), format="mp4")
            except Exception as e:
                loguru.logger.warning("wandb video upload failed at iter {}: {}", it, e)
        if media:
            wandb_run.log(media, step=it)

    def _save_rollout_video(it: int):
        nonlocal latest_rollout_video_path
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            if not warned_video_backend["done"]:
                loguru.logger.warning(
                    "rollout video skipped: `imageio` not installed in current Python env. "
                    "Install with `pip install imageio`."
                )
                warned_video_backend["done"] = True
            return

        # Snapshot training env state and RL bookkeeping, then restore after rendering rollout.
        saved_state = save_state(env.env)
        saved_rl = {
            "_sim_step": int(env._sim_step),
            "_start_sim_step": int(env._start_sim_step),
            "_episode_ctrl_steps": int(env._episode_ctrl_steps),
            "_sim_step_buf": env._sim_step_buf.clone(),
            "_start_sim_step_buf": env._start_sim_step_buf.clone(),
            "_episode_ctrl_steps_buf": env._episode_ctrl_steps_buf.clone(),
            "rew_buf": env.rew_buf.clone(),
            "reset_buf": env.reset_buf.clone(),
            "episode_length_buf": env.episode_length_buf.clone(),
            "episode_return_buf": env.episode_return_buf.clone(),
            "obs_buf": None if env.obs_buf is None else env.obs_buf.clone(),
            "extras": dict(env.extras),
        }

        frames = []
        renderer = None
        try:
            # Ensure offscreen framebuffer is large enough for requested render size.
            env.mj_model.vis.global_.offwidth = max(int(env.mj_model.vis.global_.offwidth), 720)
            env.mj_model.vis.global_.offheight = max(int(env.mj_model.vis.global_.offheight), 480)
            try:
                renderer = mujoco.Renderer(env.mj_model, height=480, width=720)
            except Exception:
                # Fallback to a conservative size if backend/model limits still reject 720x480.
                renderer = mujoco.Renderer(env.mj_model, height=360, width=640)
            mj_data_ref = mujoco.MjData(env.mj_model)
            policy = runner.alg.get_policy()
            policy_was_training = bool(getattr(policy, "training", False))
            policy.eval()

            def _run_rollout(score_only: bool, render_env_idx: int = 0):
                rollout_rewards = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
                rollout_alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
                rollout_term_sim_step = torch.full(
                    (env.num_envs,),
                    fill_value=-1,
                    device=env.device,
                    dtype=torch.long,
                )
                env._set_state_from_ref(0)
                env._sim_step = 0
                env._start_sim_step = 0
                env._episode_ctrl_steps = 0
                env._sim_step_buf.zero_()
                env._start_sim_step_buf.zero_()
                env._episode_ctrl_steps_buf.zero_()
                env.obs_buf = env._get_obs()
                local_frames: list[np.ndarray] = []
                max_ctrl_iters = (
                    None
                    if int(rollout_video_ctrl_steps) <= 0
                    else max(1, int(rollout_video_ctrl_steps))
                )
                ctrl_iter = 0

                with torch.no_grad():
                    while True:
                        if max_ctrl_iters is not None and ctrl_iter >= max_ctrl_iters:
                            break
                        obs_td = env.get_observations()
                        actions = policy(obs_td)
                        if actions.ndim == 1:
                            actions = actions.unsqueeze(0)
                        actions = torch.clamp(
                            actions.to(env.device),
                            -env.env_cfg.action_clip,
                            env.env_cfg.action_clip,
                        )

                        base_ctrl_idx = env._clamp_ref_idx(env._sim_step)
                        base_ctrl = env.ctrl_ref[base_ctrl_idx].unsqueeze(0).repeat(env.num_envs, 1)
                        ctrl = base_ctrl + actions * env.action_scale.unsqueeze(0)

                        for sub in range(env.cfg.ctrl_steps):
                            step_env(env.cfg, env.env, ctrl)
                            env._sim_step += 1
                            env._sim_step_buf += 1
                            ref_step, _ = env._ref_at(env._sim_step)
                            rew, _ = get_reward(env.cfg, env.env, ref_step)
                            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
                            rollout_rewards += (rew * rollout_alive.to(rew.dtype)) / max(1, env.cfg.ctrl_steps)

                            term = get_terminate(env.cfg, env.env, ref_step)
                            if term.any():
                                term_rew, _ = get_terminal_reward(env.cfg, env.env, ref_step)
                                term_rew = torch.nan_to_num(term_rew, nan=0.0, posinf=0.0, neginf=0.0)
                                newly_term = term & rollout_alive
                                rollout_rewards += (term_rew * newly_term.to(term_rew.dtype)) / max(1, env.cfg.ctrl_steps)
                                rollout_term_sim_step[newly_term] = env._sim_step
                                rollout_alive[newly_term] = False

                            if not score_only:
                                qpos_all = get_qpos(env.cfg, env.env)
                                qpos_render = qpos_all[render_env_idx].detach().cpu().numpy()
                                env.mj_data.qpos[:] = qpos_render
                                env.mj_data.time = float(env._sim_step * env.cfg.sim_dt)
                                mujoco.mj_forward(env.mj_model, env.mj_data)

                                ref_idx = max(0, min(env._sim_step, env.max_ref_idx))
                                mj_data_ref.qpos[:] = env.qpos_ref[ref_idx].detach().cpu().numpy()
                                mujoco.mj_forward(env.mj_model, mj_data_ref)
                                local_frames.append(
                                    render_image(env.cfg, renderer, env.mj_model, env.mj_data, mj_data_ref)
                                )

                            if env._sim_step >= env.max_ref_idx - 1:
                                break
                            if (not score_only) and (not bool(rollout_alive[render_env_idx].item())):
                                # Stop rendering once the selected env terminates.
                                break

                        env._episode_ctrl_steps += 1
                        env._episode_ctrl_steps_buf += 1
                        env.obs_buf = env._get_obs()
                        if env._sim_step >= env.max_ref_idx - 1:
                            break
                        if not rollout_alive.any():
                            break
                        if (not score_only) and (not bool(rollout_alive[render_env_idx].item())):
                            break
                        ctrl_iter += 1

                # If never terminated, treat as surviving until the current rollout end.
                rollout_len_sim = rollout_term_sim_step.clone()
                alive_end = rollout_len_sim < 0
                rollout_len_sim[alive_end] = int(env._sim_step)
                return rollout_rewards, local_frames, rollout_len_sim

            # Pass 1: score all parallel envs, choose the highest-score rollout.
            score_rewards, _, score_lengths = _run_rollout(score_only=True)
            best_env_idx = int(torch.argmax(score_rewards).item())
            best_len = int(score_lengths[best_env_idx].item())
            best_reward = float(score_rewards[best_env_idx].item())
            loguru.logger.info(
                "[RL][viz] iter={} best-score rollout env={} sim_step={} reward={:.4f}",
                it,
                best_env_idx,
                best_len,
                best_reward,
            )

            # Pass 2: rerun and render only the best env (much cheaper than rendering all envs).
            _, frames, _ = _run_rollout(score_only=False, render_env_idx=best_env_idx)

            if frames:
                video_path = viz_dir / f"rollout_iter_{it}.mp4"
                imageio.mimsave(str(video_path), frames, fps=max(1, int(round(1.0 / env.cfg.sim_dt))))
                latest_rollout_video_path = video_path
                if it == 0:
                    imageio.mimsave(str(viz_dir / "rollout_iter_0.mp4"), frames, fps=max(1, int(round(1.0 / env.cfg.sim_dt))))
        except Exception as e:
            loguru.logger.warning("rollout video save failed at iter {}: {}", it, e)
        finally:
            try:
                if "policy" in locals():
                    policy.train(policy_was_training)
            except Exception:
                pass
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
                renderer = None
            load_state(env.env, saved_state)
            env._sim_step = saved_rl["_sim_step"]
            env._start_sim_step = saved_rl["_start_sim_step"]
            env._episode_ctrl_steps = saved_rl["_episode_ctrl_steps"]
            env._sim_step_buf = saved_rl["_sim_step_buf"]
            env._start_sim_step_buf = saved_rl["_start_sim_step_buf"]
            env._episode_ctrl_steps_buf = saved_rl["_episode_ctrl_steps_buf"]
            env.rew_buf = saved_rl["rew_buf"]
            env.reset_buf = saved_rl["reset_buf"]
            env.episode_length_buf = saved_rl["episode_length_buf"]
            env.episode_return_buf = saved_rl["episode_return_buf"]
            env.obs_buf = saved_rl["obs_buf"]
            env.extras = saved_rl["extras"]
            qpos_restore = get_qpos(env.cfg, env.env)[0].detach().cpu().numpy()
            env.mj_data.qpos[:] = qpos_restore
            env.mj_data.time = float(env._sim_step * env.cfg.sim_dt)
            mujoco.mj_forward(env.mj_model, env.mj_data)

    def _save_viz_snapshot(it: int):
        payload = {"history": history}
        (viz_dir / "reward_history.json").write_text(json.dumps(payload, indent=2))
        try:
            import matplotlib.pyplot as plt  # type: ignore

            xs = [h["iter"] for h in history]
            ys = [h["mean_reward"] for h in history]
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(xs, ys, linewidth=2.0)
            ax.set_ylabel("Mean Reward")
            ax.set_xlabel("Learning Iteration")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(viz_dir / "reward_curve_latest.png", dpi=140)
            plt.close(fig)
        except Exception:
            # Matplotlib is optional; JSON snapshots are always written.
            pass

    def wrapped_log(*args, **kwargs):
        ep_extras_before = list(runner.logger.ep_extras)
        orig_log(*args, **kwargs)
        # rsl_rl passes `it` as kwarg; keep a fallback for positional usage.
        it = int(kwargs.get("it", args[0] if args else runner.current_learning_iteration))
        # Use rsl_rl's own training buffers as the primary metric source.
        train_rew_mean = (
            float(statistics.mean(runner.logger.rewbuffer))
            if len(runner.logger.rewbuffer) > 0
            else float("nan")
        )
        # Keep extras-derived values only as debug references; they are not the PPO train metric.
        extras_r_mean = float("nan")
        if ep_extras_before:
            rs = []
            for ep in ep_extras_before:
                if "r" in ep:
                    rs.append(float(ep["r"]))
                elif "/episode_reward" in ep:
                    rs.append(float(ep["/episode_reward"]))
            if rs:
                extras_r_mean = float(sum(rs) / len(rs))

        history.append(
            {
                "iter": float(it),
                "mean_reward": train_rew_mean,
                "extras_r_mean": extras_r_mean,
            }
        )
        loguru.logger.info(
            "[RL] iter={} train_mean_reward={} (extras_r={})",
            it,
            f"{train_rew_mean:.4f}" if train_rew_mean == train_rew_mean else "nan",
            f"{extras_r_mean:.4f}" if extras_r_mean == extras_r_mean else "nan",
        )
        # Refresh curve/png every iteration so W&B image updates each epoch.
        _save_viz_snapshot(it)
        # Upload curve every iteration (epoch), video only on viz iterations.
        _wandb_upload(it, upload_video=False)
        if it == 0 or (it + 1) % int(viz_interval) == 0:
            _save_rollout_video(it)
            _wandb_upload(it, upload_video=True)

    runner.logger.log = wrapped_log
    return history


def main(args: RLTrainArgs):
    cfg = _make_spider_config(args)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(cfg, cfg.data_path)
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    env = MJWPRslResidualEnv(
        cfg,
        ref_data,
        MJWPRLEnvCfg(
            future_ctrl_steps=args.future_ctrl_steps,
            max_episode_ctrl_steps=args.max_episode_ctrl_steps,
        ),
    )

    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rsl_rl.runners import OnPolicyRunner  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "rsl_rl is not installed. Install it in your environment, then rerun.\n"
            "Example (inside repo venv): `pip install rsl-rl-lib` or your preferred rsl_rl package source."
        ) from e
    wandb_run = None
    if args.wandb_media_only:
        try:
            import wandb  # type: ignore

            wandb_kwargs = {
                "project": args.wandb_project,
                "name": args.wandb_run_name or args.experiment_name,
                "save_code": False,
                "config": {},
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            wandb_run = wandb.init(**wandb_kwargs)
            loguru.logger.info("W&B media-only upload enabled: project={}, run={}", args.wandb_project, args.wandb_run_name or args.experiment_name)
        except Exception as e:
            loguru.logger.warning("Failed to initialize wandb media-only uploader: {}", e)

    train_cfg = _make_rsl_train_cfg(args)
    loguru.logger.info(
        "Starting RL residual training (single task / single demo) with num_envs={}, future_ctrl_steps={}, data_path={}",
        args.num_envs,
        args.future_ctrl_steps,
        cfg.data_path,
    )
    loguru.logger.info(
        "Observation design: proprio(real qpos/qvel) + tool state(raw pos+quat) + goal(ref_ctrl); action = ctrl_ref[t] + residual",
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=cfg.device)
    reward_history = _install_training_viz_hook(
        runner,
        env,
        log_dir,
        args.viz_interval,
        args.rollout_video_ctrl_steps,
        wandb_run=wandb_run,
    )
    # rsl_rl runner API commonly supports `learn(num_learning_iterations=...)`.
    runner.learn(num_learning_iterations=args.learning_iterations, init_at_random_ep_len=False)
    if args.viz_interval > 0 and reward_history:
        loguru.logger.info("Saved RL reward visualizations to {}", log_dir / "viz")
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main(tyro.cli(RLTrainArgs))
