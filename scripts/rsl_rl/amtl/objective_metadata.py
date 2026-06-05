from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveMetadata:
    group_name: str
    is_penalty: bool


OBJECTIVE_METADATA: dict[str, ObjectiveMetadata] = {
    "motion_global_anchor_pos": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_global_anchor_ori": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_body_lin_vel": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_end_effector_lin_vel": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_body_ang_vel": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_body_pos": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_body_ori": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_foot_ori": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_foot_pos": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_foot_ang_vel": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_foot_contact_match": ObjectiveMetadata(group_name="Tracking", is_penalty=False),
    "motion_hand_ori": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_hand_pos": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_trunk_ori": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_trunk_pos": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "motion_trunk_ang_vel": ObjectiveMetadata(group_name="Fidelity", is_penalty=False),
    "amp_style": ObjectiveMetadata(group_name="Style", is_penalty=False),
    "action_rate_l2": ObjectiveMetadata(group_name="Smoothness", is_penalty=True),
    "joint_limit": ObjectiveMetadata(group_name="Survival", is_penalty=True),
    "undesired_contacts": ObjectiveMetadata(group_name="Survival", is_penalty=True),
}


def get_objective_metadata(objective_name: str) -> ObjectiveMetadata:
    if objective_name in OBJECTIVE_METADATA:
        return OBJECTIVE_METADATA[objective_name]

    if any(token in objective_name for token in ("power", "energy", "torque")):
        return ObjectiveMetadata(group_name="Power", is_penalty=True)

    if objective_name.startswith("motion_"):
        return ObjectiveMetadata(group_name="Tracking", is_penalty=False)

    if any(token in objective_name for token in ("contact", "limit", "fall", "terminate", "collision")):
        return ObjectiveMetadata(group_name="Survival", is_penalty=True)

    if any(token in objective_name for token in ("action_rate", "jerk", "smooth")):
        return ObjectiveMetadata(group_name="Smoothness", is_penalty=True)

    return ObjectiveMetadata(group_name="Tracking", is_penalty=False)
