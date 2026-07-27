from __future__ import annotations

import numpy as np
import os
import uuid
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import \
    convert_absolute_to_relative_se2_array
from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch


def get_local_ego_poses(scenario):
    thread_id = str(uuid.uuid4())
    print(
        f"Processing scenario in thread_id={thread_id}"
    )
    init_ego_state = scenario.initial_ego_state
    future_traj = scenario.get_ego_future_trajectory(0, 4)
    local_ego_poses = convert_absolute_to_relative_se2_array(
        init_ego_state.rear_axle, np.array([tmp.rear_axle.serialize() for tmp in future_traj], dtype=np.float64)
    )
    return local_ego_poses[None].astype(np.float32)


def main():
    root = 'your_nuplan_db_root'
    save_dir = 'your_tmp_save_dir'

    split = 'trainval'
    logs = os.listdir(f'{root}/nuplan/nuplan-v1.1/splits/{split}')
    logs = [tmp.replace('.db', '') for tmp in logs if tmp.endswith('.db')]
    start_idx = 0
    end_idx = 700000
    os.makedirs(save_dir, exist_ok=True)
    save_file = f'{save_dir}/{split}-700k-rear-axle.npy'

    print(f'total logs: {len(logs)}')
    filter = ScenarioFilter(
        None, None,
        logs,
        None, None, None, None, None, False, False, False
    )
    worker = RayDistributedNoTorch(threads_per_node=16)

    builder = NuPlanScenarioBuilder(
        data_root=f'{root}/nuplan/',
        map_root=f'{root}/nuplan/maps',
        sensor_root=f'{root}/nuplan/',
        db_files=f'{root}/nuplan/nuplan-v1.1/splits/{split}',
        map_version='nuplan-maps-v1.0',
        scenario_mapping=ScenarioMapping({}, 0.5)
    )
    scenarios = builder.get_scenarios(filter, worker)

    print(f'total scenarios: {len(scenarios)}, now: {start_idx} to {end_idx}')
    all_ego_poses = worker_map(worker, get_local_ego_poses, scenarios[start_idx:end_idx])

    all_ego_poses = np.concatenate(all_ego_poses, axis=0)
    print(f'save to: {save_file}')
    np.save(save_file, all_ego_poses)
    print(all_ego_poses.shape)


if __name__ == '__main__':
    main()
