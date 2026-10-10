// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { SchemaRobotSimulation } from '../../api/openapi-spec';

export type RobotSimulation = SchemaRobotSimulation;

/** What a simulated robot's simulation serves (viewer, speed, cameras), polled while shown. */
export const useRobotSimulation = (project_id: string, robot_id: string, enabled = true) => {
    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/simulation',
        { params: { path: { project_id, robot_id } } },
        { enabled, refetchInterval: enabled ? 2_000 : false }
    );
};
