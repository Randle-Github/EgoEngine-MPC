from __future__ import annotations

from dataclasses import dataclass

import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from spider.config import Config
from spider.simulators.mjwp import (
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    load_env_params,
    setup_env,
    setup_mj_model,
    step_env,
    sync_env_mujoco,
)


def _quat_wxyz_to_rotmat_torch(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) to rotation matrix. quat shape (..., 4)."""
    w, x, y, z = quat.unbind(dim=-1)
    two = torch.tensor(2.0, device=quat.device, dtype=quat.dtype)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - two * (yy + zz)
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)
    r10 = two * (xy + wz)
    r11 = 1 - two * (xx + zz)
    r12 = two * (yz - wx)
    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = 1 - two * (xx + yy)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = _quat_wxyz_to_rotmat_torch(quat_wxyz)
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def _build_action_scale(config: Config) -> torch.Tensor:
    """Use MPC noise component scales as residual action scales."""
    scale = torch.ones(config.nu, device=config.device, dtype=torch.float32)
    if config.embodiment_type in ["bimanual", "right", "left"]:
        scale[:3] = float(config.pos_noise_scale)
        scale[3:6] = float(config.rot_noise_scale)
        if config.embodiment_type == "bimanual":
            half = config.nu // 2
            scale[6:half] = float(config.joint_noise_scale)
            scale[half : half + 3] = float(config.pos_noise_scale)
            scale[half + 3 : half + 6] = float(config.rot_noise_scale)
            scale[half + 6 :] = float(config.joint_noise_scale)
        else:
            scale[6:] = float(config.joint_noise_scale)
    else:
        scale[:] = float(config.joint_noise_scale)
    # Match MPC trajectory sampling magnitude roughly.
    scale *= float(config.first_ctrl_noise_scale)
    return scale


@dataclass
class MJWPRLEnvCfg:
    future_ctrl_steps: int = 0
    max_episode_ctrl_steps: int = 64
    action_clip: float = 1.0


class MJWPRslResidualEnv:
    """Single-task residual RL env on top of MJWP.

    Parallel residual RL env on top of MJWP.

    Each world keeps its own reference phase / episode clock, which is important
    for PPO sample diversity and to avoid "one world terminates => all reset".
    """

    def __init__(
        self,
        config: Config,
        ref_data: tuple[torch.Tensor, ...],
        env_cfg: MJWPRLEnvCfg,
    ) -> None:
        self.cfg = config
        self.ref_data = ref_data
        self.env_cfg = env_cfg
        self.device = config.device
        self.num_envs = int(config.num_samples)
        self.num_actions = int(config.nu)
        self.num_privileged_obs = 0
        self.max_episode_length = int(env_cfg.max_episode_ctrl_steps)
        self.action_scale = _build_action_scale(config)

        self.env = setup_env(config, ref_data)
        self.mj_model = setup_mj_model(config)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.qpos_ref, self.qvel_ref, self.ctrl_ref, self.contact_ref, self.contact_pos_ref = (
            ref_data
        )
        self.max_ref_idx = int(self.qpos_ref.shape[0] - 1)
        self.nq_obj = int(config.nq_obj)
        self.nv_obj = 12 if config.embodiment_type == "bimanual" else 6

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_return_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.obs_buf: torch.Tensor | None = None
        self.extras: dict = {}

        self._sim_step = 0
        self._start_sim_step = 0
        self._episode_ctrl_steps = 0
        self._sim_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._start_sim_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._episode_ctrl_steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._dr_env_params = self._build_dr_env_params()

        self.obs_buf = self._reset_all()
        self.num_obs = int(self.obs_buf.shape[1])

    def _build_dr_env_params(self) -> list[dict]:
        # Mirror the DR parameter grid construction used in run_mjwp.py.
        if int(self.cfg.num_dr) == 0:
            xy_offset_list = [0.0]
            pair_margin_list = [0.0]
        else:
            xy_offset_list = np.linspace(
                float(self.cfg.xy_offset_range[0]),
                float(self.cfg.xy_offset_range[1]),
                int(self.cfg.num_dr),
            )
            pair_margin_list = np.linspace(
                float(self.cfg.pair_margin_range[0]),
                float(self.cfg.pair_margin_range[1]),
                int(self.cfg.num_dr),
            )
        return [
            {"xy_offset": float(xy_offset), "pair_margin": float(pair_margin)}
            for xy_offset, pair_margin in zip(xy_offset_list, pair_margin_list)
        ]

    def _sample_and_apply_dr(self) -> None:
        if not self._dr_env_params:
            return
        env_param = self._dr_env_params[np.random.randint(0, len(self._dr_env_params))]
        load_env_params(self.cfg, self.env, env_param)

    def _clamp_ref_idx(self, idx: int) -> int:
        return int(max(0, min(idx, self.max_ref_idx)))

    def _ref_at(self, sim_step: int):
        idx = self._clamp_ref_idx(sim_step)
        return tuple(r[idx] for r in self.ref_data), idx

    def _ref_at_indices(self, sim_steps: torch.Tensor):
        idx = torch.clamp(sim_steps.to(torch.long), 0, self.max_ref_idx)
        ref = tuple(r.index_select(0, idx) for r in self.ref_data)
        return ref, idx

    def _set_state_from_ref(self, sim_step: int) -> None:
        idx = self._clamp_ref_idx(sim_step)
        qpos_np = self.qpos_ref[idx].detach().cpu().numpy()
        qvel_np = self.qvel_ref[idx].detach().cpu().numpy()
        ctrl_np = self.ctrl_ref[idx].detach().cpu().numpy()
        self.mj_data.qpos[:] = qpos_np
        self.mj_data.qvel[:] = qvel_np
        if self.mj_model.nu > 0:
            self.mj_data.ctrl[:] = ctrl_np
        self.mj_data.time = idx * float(self.cfg.sim_dt)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        sync_env_mujoco(self.cfg, self.env, self.mj_data)
        self._sim_step = idx
        self._start_sim_step = idx
        self._episode_ctrl_steps = 0
        self._sim_step_buf.fill_(idx)
        self._start_sim_step_buf.fill_(idx)
        self._episode_ctrl_steps_buf.zero_()

    def _refresh_mj_data_from_env0(self) -> None:
        qpos0 = get_qpos(self.cfg, self.env)[0].detach().cpu().numpy()
        qvel0 = get_qvel(self.cfg, self.env)[0].detach().cpu().numpy()
        self.mj_data.qpos[:] = qpos0
        self.mj_data.qvel[:] = qvel0
        self.mj_data.time = float(int(self._sim_step_buf[0].item()) * self.cfg.sim_dt)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _set_done_envs_from_ref(self, done_mask: torch.Tensor, sim_steps: torch.Tensor) -> None:
        if not bool(done_mask.any()):
            return
        done_mask = done_mask.to(torch.bool)
        idx = torch.clamp(sim_steps.to(torch.long), 0, self.max_ref_idx)
        qpos_all = get_qpos(self.cfg, self.env).detach().clone()
        qvel_all = get_qvel(self.cfg, self.env).detach().clone()
        qpos_all[done_mask] = self.qpos_ref.index_select(0, idx[done_mask])
        qvel_all[done_mask] = self.qvel_ref.index_select(0, idx[done_mask])
        wp.copy(self.env.data_wp.qpos, wp.from_torch(qpos_all))
        wp.copy(self.env.data_wp.qvel, wp.from_torch(qvel_all))
        if hasattr(self.env.data_wp, "ctrl"):
            ctrl_all = wp.to_torch(self.env.data_wp.ctrl).clone()
            ctrl_all[done_mask] = self.ctrl_ref.index_select(0, idx[done_mask])
            wp.copy(self.env.data_wp.ctrl, wp.from_torch(ctrl_all))
        if hasattr(self.env.data_wp, "time"):
            time_all = wp.to_torch(self.env.data_wp.time).clone()
            time_all[done_mask] = idx[done_mask].to(torch.float32) * float(self.cfg.sim_dt)
            wp.copy(self.env.data_wp.time, wp.from_torch(time_all))
        self._refresh_mj_data_from_env0()

    def _sync_scalar_clocks_from_env0(self) -> None:
        self._sim_step = int(self._sim_step_buf[0].item())
        self._start_sim_step = int(self._start_sim_step_buf[0].item())
        self._episode_ctrl_steps = int(self._episode_ctrl_steps_buf[0].item())

    def _reset_all(self) -> torch.Tensor:
        max_start = max(
            0,
            self.max_ref_idx
            - self.cfg.horizon_steps
            - self.cfg.ctrl_steps
            - self.env_cfg.max_episode_ctrl_steps * self.cfg.ctrl_steps
            - 2,
        )
        if max_start > 0:
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            starts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._start_sim_step_buf = starts.clone()
        self._sim_step_buf = starts.clone()
        self._episode_ctrl_steps_buf.zero_()
        self._set_done_envs_from_ref(
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
            self._sim_step_buf,
        )
        self._sample_and_apply_dr()
        self._sync_scalar_clocks_from_env0()
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        self.episode_return_buf.zero_()
        return self._get_obs()

    def _reset_done(self, done_mask: torch.Tensor) -> None:
        if not bool(done_mask.any()):
            return
        max_start = max(
            0,
            self.max_ref_idx
            - self.cfg.horizon_steps
            - self.cfg.ctrl_steps
            - self.env_cfg.max_episode_ctrl_steps * self.cfg.ctrl_steps
            - 2,
        )
        n_done = int(done_mask.sum().item())
        if max_start > 0:
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(n_done,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            starts = torch.zeros(n_done, device=self.device, dtype=torch.long)
        self._start_sim_step_buf[done_mask] = starts
        self._sim_step_buf[done_mask] = starts
        self._episode_ctrl_steps_buf[done_mask] = 0
        self.episode_length_buf[done_mask] = 0
        self.episode_return_buf[done_mask] = 0.0
        self.rew_buf[done_mask] = 0.0
        self.reset_buf[done_mask] = False
        self._set_done_envs_from_ref(done_mask, self._sim_step_buf)
        self._sync_scalar_clocks_from_env0()

    def _get_obs(self) -> torch.Tensor:
        qpos = get_qpos(self.cfg, self.env).detach()
        qvel = get_qvel(self.cfg, self.env).detach()
        robot_qpos = qpos[:, : -self.nq_obj]
        robot_qvel = qvel[:, : -self.nv_obj]
        obj_qpos = qpos[:, -self.nq_obj :]
        ref_now, idx_now = self._ref_at_indices(self._sim_step_buf)
        ref_ctrl = ref_now[2]

        # Keep object state in raw simulator/reference state format (pos + quat for single-hand).
        # This matches the user's requested "real states" style observation.
        obj_feat = obj_qpos
        _ = idx_now  # kept for debugging parity

        obs = torch.cat(
            [
                robot_qpos,    # floating hand proprioception (real qpos state)
                robot_qvel,    # floating hand proprioception (real qvel state)
                obj_feat,      # object/tool state (raw qpos state: pos + quat)
                ref_ctrl,      # goal command from reference trajectory (current ctrl)
            ],
            dim=-1,
        ).to(torch.float32)
        return obs

    def _obs_tensordict(self) -> TensorDict:
        assert self.obs_buf is not None
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self._obs_tensordict()

    def reset(self):
        self.obs_buf = self._reset_all()
        return self._obs_tensordict()

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = torch.clamp(actions, -self.env_cfg.action_clip, self.env_cfg.action_clip)

        total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        term_reason_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        base_ctrl_idx = torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)
        base_ctrl = self.ctrl_ref.index_select(0, base_ctrl_idx)
        ctrl = base_ctrl + actions * self.action_scale.unsqueeze(0)

        for sub in range(self.cfg.ctrl_steps):
            step_env(self.cfg, self.env, ctrl)
            self._sim_step_buf += 1
            ref_step, _ = self._ref_at_indices(self._sim_step_buf)
            rew, _ = get_reward(self.cfg, self.env, ref_step)
            # NaN/Inf safety: prevent invalid rewards from contaminating PPO updates.
            if not torch.isfinite(rew).all():
                bad = ~torch.isfinite(rew)
                rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
                done[bad] = True
                term_reason_id_buf[bad] = 1.0
            total_rew += rew
            if sub == self.cfg.ctrl_steps - 1:
                term_rew, _ = get_terminal_reward(self.cfg, self.env, ref_step)
                total_rew += term_rew
        # Per-env terminate on final state of this control step (keeps all worlds stepped equally).
        term = get_terminate(self.cfg, self.env, ref_step)
        if term.any():
            done |= term
            term_reason_id_buf[term] = 1.0

        self._episode_ctrl_steps_buf += 1
        self.episode_length_buf += 1
        timeout = self._episode_ctrl_steps_buf >= self.max_episode_length
        out_of_ref = self._sim_step_buf >= (self.max_ref_idx - 2)
        newly_timeout = timeout & ~done
        newly_ref_end = out_of_ref & ~done
        done |= timeout | out_of_ref
        term_reason_id_buf[newly_timeout] = 2.0
        term_reason_id_buf[newly_ref_end] = 3.0

        self.rew_buf = total_rew / max(1, self.cfg.ctrl_steps)
        if not torch.isfinite(self.rew_buf).all():
            bad = ~torch.isfinite(self.rew_buf)
            self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
            done[bad] = True
            term_reason_id_buf[bad] = 1.0
        self.reset_buf = done.clone()
        self.episode_return_buf += self.rew_buf
        # Preserve outputs before any reset path mutates internal buffers.
        rew_out = self.rew_buf.clone()
        done_out = done.clone()

        episode_info = {}
        if done.any():
            done_mask = done.clone()
            episode_info = {
                "r": float(self.episode_return_buf[done_mask].mean().item()),
                "l": float(self._episode_ctrl_steps_buf[done_mask].to(torch.float32).mean().item()),
                "sim_step": float(self._sim_step_buf[done_mask].to(torch.float32).mean().item()),
                "term_reason_id": float(term_reason_id_buf[done_mask].mean().item()),
            }

        if done.any():
            self._reset_done(done)
            next_obs = self._get_obs()
        else:
            next_obs = self._get_obs()
        if not torch.isfinite(next_obs).all():
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.obs_buf = next_obs
        self._sync_scalar_clocks_from_env0()

        # Preserve outputs before _reset_all() clears internal buffers.
        time_outs = (term_reason_id_buf == 2.0) | (term_reason_id_buf == 3.0)
        if episode_info:
            self.extras = {
                "time_outs": time_outs,
                "log": {
                    "/episode_reward": float(episode_info["r"]),
                    "/episode_length": float(episode_info["l"]),
                    "/sim_step": float(episode_info["sim_step"]),
                    "/term_reason_id": float(episode_info["term_reason_id"]),
                },
                "episode": episode_info,
            }
        else:
            self.extras = {"time_outs": time_outs}
        return self._obs_tensordict(), rew_out, done_out, self.extras

    def close(self):
        return None
