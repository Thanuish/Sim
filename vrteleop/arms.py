"""The two teleoperated UR5e arms: controller input -> clutch mapping -> differential IK ->
joint and gripper commands. Knows nothing about cloth, recording or the network."""
from __future__ import annotations

import mujoco
import numpy as np

from . import scene as S
from .ik import DiffIK
from .teleop import ArmTeleop


class DualArms:
    def __init__(self, info: S.SceneInfo, task: str, pos_scale: float = 1.0):
        self.info = info
        m = info.model
        self.ik = {s: DiffIK(m, info.arm_qpos_adr[s], info.arm_dof_adr[s], info.ee_site[s], S.HOME_Q[s])
                   for s in S.SIDES}
        self.teleop = {s: ArmTeleop(pos_scale=pos_scale, z_min=S.PINCH_Z_MIN[task]) for s in S.SIDES}
        self.q_cmd = {s: S.HOME_Q[s].copy() for s in S.SIDES}

    def reset(self, d: mujoco.MjData):
        for s in S.SIDES:
            self.q_cmd[s] = S.HOME_Q[s].copy()
            pos, quat = self.ik[s].fk(self.q_cmd[s], d.qpos)
            self.teleop[s].reset(pos, quat)

    def control(self, d: mujoco.MjData, inputs: dict | None, dt: float):
        """One control tick. inputs: the headset message ({'left': {...}, 'right': {...}}), or
        None when it is stale (the arms disengage and hold still)."""
        for s in S.SIDES:
            tel = self.teleop[s]
            if not tel.engaged:
                # keep the virtual target glued to the commanded pose while idle
                tel.target_pos, tel.target_quat = self.ik[s].fk(self.q_cmd[s])
            if inputs is not None:
                tel.update(inputs.get(s), dt)
            else:
                tel.disengage()
            if tel.engaged:
                self.q_cmd[s] = self.ik[s].step(self.q_cmd[s], tel.target_pos, tel.target_quat, dt)
            d.ctrl[self.info.arm_act[s]] = self.q_cmd[s]
            d.ctrl[self.info.grip_act[s]] = 255.0 * tel.gripper

    def gripper(self, side: str) -> float:
        """Commanded closure, 0 open .. 1 closed."""
        return self.teleop[side].gripper

    def engaged(self, side: str) -> bool:
        return self.teleop[side].engaged

    def ee_pose(self, d: mujoco.MjData, side: str):
        sid = self.info.ee_site[side]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[sid])
        return d.site_xpos[sid].copy(), q
