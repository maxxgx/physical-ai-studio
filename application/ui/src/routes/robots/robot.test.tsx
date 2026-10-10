// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { Robot } from './robot';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';
const robot = {
    id: ROBOT_ID,
    name: 'Sim arm',
    type: 'SO101_Follower' as const,
    payload: { connection_string: '', serial_number: '' },
};

const mockSimulatedRobot = (connection_status: 'online' | 'offline', simulation: object) => {
    server.use(
        http.get('/api/projects/{project_id}/robots/online', () =>
            HttpResponse.json([{ ...robot, connection_status }])
        ),
        // After /online: `{robot_id}` would match it too.
        http.get('/api/projects/{project_id}/robots/{robot_id}', () => HttpResponse.json(robot)),
        http.get('/api/simulation', () => HttpResponse.json({ cameras: [], ...simulation } as never)),
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: 'SO101_Follower',
                    display_name: 'SO101 Follower',
                    role: 'follower',
                    urdf_path: '',
                    package_map: {},
                    joint_map: {},
                    zero_calibration: null,
                    simulation: {
                        scenes: [{ id: 'single_pick_place', display_name: 'Pick & place', description: '' }],
                        default_scene: 'single_pick_place',
                        max_seed: null,
                        labels: [],
                    },
                },
            ])
        )
    );
};

const renderRobotPage = () =>
    render(<Robot />, {
        route: `/projects/${PROJECT_ID}/robots/${ROBOT_ID}`,
        path: '/projects/:project_id/robots/:robot_id',
    });

describe('Robot page of a simulated robot', () => {
    it('offers to start the simulation when it is not running', async () => {
        mockSimulatedRobot('offline', { status: 'idle' });
        renderRobotPage();

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Start simulation…' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    });

    it('offers the progress while Studio starts the simulation', async () => {
        mockSimulatedRobot('offline', { status: 'starting', robot_id: ROBOT_ID });
        renderRobotPage();

        expect(await screen.findByRole('heading', { name: 'Simulation starting' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Show progress…' })).toBeInTheDocument();
    });

    it('shows the robot when its simulation runs', async () => {
        mockSimulatedRobot('online', { status: 'idle' });
        renderRobotPage();

        expect(await screen.findByRole('button', { name: 'Identify' })).toBeInTheDocument();
        expect(screen.queryByRole('heading', { name: 'Simulation not running' })).not.toBeInTheDocument();
    });
});
