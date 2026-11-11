// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { $api } from '../../api/client';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { render } from '../../test-utils/render';
import { SessionViewer } from './session-viewer';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';

rs.mock('dockview-react', () => ({ DockviewReact: () => <div>Dock</div> }));

rs.mock('./workspace/part-controls', () => ({ usePartControls: () => ({ pendingPart: null, act: rs.fn() }) }));

const retry = rs.fn();
/** What the mocked session holds: by default an environment that loaded, failed. */
const session: { loaded: boolean; error: { message: string; error_code: string } | null } = {
    loaded: true,
    error: null,
};

rs.mock('./runtime-session-provider', () => ({
    failureOf: (statuses: Record<string, { state: string; error?: string }>, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'failed') : null,
    useRuntimeSession: () => ({
        draft: { name: 'Bench', robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }], cameras: [] },
        state: {
            environment: session.loaded
                ? { environment_id: null, robot_id: ROBOT_ID, leader_id: null, cameras: [] }
                : null,
        },
        error: session.error,
        loading: null,
        statuses: session.loaded
            ? {
                  environment: {
                      type: 'status',
                      component: 'environment',
                      state: 'failed',
                      error: 'follower: not running',
                  },
              }
            : {},
        retry,
    }),
}));

const mockFollower = (simulated: boolean) =>
    server.use(
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
                    simulation: simulated
                        ? {
                              scenes: [{ id: 'single_pick_place', display_name: 'Pick & place', description: '' }],
                              default_scene: 'single_pick_place',
                              max_seed: null,
                              labels: ['1 arm'],
                          }
                        : null,
                },
            ])
        ),
        http.get(
            '/api/projects/{project_id}/robots/{robot_id}',
            () =>
                HttpResponse.json({
                    id: ROBOT_ID,
                    name: 'Arm',
                    type: 'SO101_Follower',
                    payload: { connection_string: '', serial_number: '' },
                }) as never
        )
    );

const renderViewer = (queryClient = createQueryClient()) =>
    render(<SessionViewer cancelHref='/' />, {
        route: `/projects/${PROJECT_ID}/record`,
        path: '/projects/:project_id/record',
        queryClient,
    });

beforeEach(() => {
    retry.mockClear();
    session.loaded = true;
    session.error = null;
});

describe('SessionViewer when the simulation of its cameras does not run', () => {
    const refuseLoading = () => {
        session.loaded = false;
        session.error = {
            message: "Start the simulation of 'Arm' to use its camera 'Arm wrist'.",
            error_code: 'simulation_not_running',
        };
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
                HttpResponse.json({ running: false, owner_name: 'arm' } as never)
            ),
            http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] } as never))
        );
    };

    it('offers to start the simulation of the page follower', async () => {
        mockFollower(true);
        refuseLoading();
        renderViewer();

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        await userEvent.setup().click(screen.getByRole('button', { name: 'Start simulation…' }));
        expect(await screen.findByRole('dialog')).toBeInTheDocument();
    });

    it('loads the environment again once the simulation runs', async () => {
        mockFollower(true);
        refuseLoading();
        const queryClient = createQueryClient();
        renderViewer(queryClient);
        await screen.findByRole('heading', { name: 'Simulation not running' });
        expect(retry).not.toHaveBeenCalled();

        act(() => {
            queryClient.setQueryData(
                $api.queryOptions('get', '/api/projects/{project_id}/robots/{robot_id}/simulation', {
                    params: { path: { project_id: PROJECT_ID, robot_id: ROBOT_ID } },
                }).queryKey,
                { running: true, started_by: 'studio', owner_name: 'arm', cameras: [] } as never
            );
        });

        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));
    });

    it('offers nothing for a follower without a simulation', async () => {
        mockFollower(false);
        refuseLoading();
        renderViewer();

        expect(await screen.findByText(/Start the simulation of 'Arm'/)).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Start simulation…' })).not.toBeInTheDocument();
    });
});

describe('SessionViewer when the first load fails', () => {
    it('shows the dock for a simulated follower, whose cell offers to start the simulation', async () => {
        mockFollower(true);
        renderViewer();

        expect(await screen.findByText('Dock')).toBeInTheDocument();
    });

    it('keeps showing the failure for any other follower', async () => {
        mockFollower(false);
        renderViewer();

        expect(await screen.findByText('follower: not running')).toBeInTheDocument();
        expect(screen.queryByText('Dock')).not.toBeInTheDocument();
    });
});
