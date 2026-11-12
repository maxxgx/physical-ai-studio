// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { describeAutopilot } from '../../simulation/autopilot-status';
import * as actualProvider from '../runtime-session-provider' with { rstest: 'importActual' };
import { RecordMode } from './record-mode';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';
const ENVIRONMENT_ID = 'environment-1';

const send = rs.fn(async () => ({ type: 'ack', request_id: '', ok: true }));
const session = { autopilot: false };
let recording: { phase: string; message: string; saved: number; discarded: number; max_episodes: number } | null = null;

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    savedEnvironmentDraft: actualProvider.savedEnvironmentDraft,
    useRuntimeSession: () => ({
        state: {
            environment: {
                environment_id: ENVIRONMENT_ID,
                name: 'Sim bench',
                robot_id: ROBOT_ID,
                leader_id: session.autopilot ? 'autopilot-leader' : null,
                autopilot: session.autopilot,
                joint_names: [],
                cameras: [],
            },
            control: null,
            dataset: null,
        },
        statuses: {},
        send,
    }),
}));

const autopilot = {
    available: true,
    unavailable_reason: null,
    mode: 'leader',
    phase: 'approach',
    target: 'cpu',
    attempt: 1,
    max_attempts: 2,
    outcome: 'running',
    failure: null,
};

beforeEach(() => {
    send.mockClear();
    session.autopilot = false;
    recording = null;
    server.use(
        http.get('/api/settings', () => HttpResponse.json({ hotkeys: { bindings: {} } } as never)),
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: 'MuJoCo_SO101_Follower',
                    display_name: 'MuJoCo SO101 Follower',
                    role: 'follower',
                    urdf_path: '',
                    package_map: {},
                    joint_map: {},
                    simulation: {
                        scenes: [{ id: 'conveyor', display_name: 'Conveyor', description: '' }],
                        default_scene: 'conveyor',
                    },
                },
            ] as never)
        ),
        http.get('/api/projects/{project_id}/robots', () =>
            HttpResponse.json([{ id: ROBOT_ID, name: 'Sim arm', type: 'MuJoCo_SO101_Follower', payload: {} }] as never)
        ),
        http.get('/api/projects/{project_id}/environments', () =>
            HttpResponse.json([
                {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }],
                    cameras: [{ camera_id: 'camera-1' }],
                },
            ] as never)
        ),
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({
                running: true,
                started_by: 'studio',
                owner_name: 'arm',
                cameras: [],
                autopilot,
                recording,
            } as never)
        )
    );
});

const renderMode = () =>
    render(<RecordMode />, { route: `/projects/${PROJECT_ID}/record`, path: '/projects/:project_id/record' });

describe('RecordMode with a simulated follower', () => {
    it('lets the autopilot lead this session without editing the saved environment', async () => {
        renderMode();
        const user = userEvent.setup();

        await user.click(await screen.findByRole('button', { name: /Teleoperator/ }));
        await user.click(await screen.findByRole('option', { name: 'Autopilot (simulation)' }));

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'load_environment',
                environment: {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'autopilot' } }],
                    cameras: [{ camera_id: 'camera-1' }],
                },
                unload_incompatible: false,
            })
        );
    });

    it('shows what the autopilot does while it leads', async () => {
        session.autopilot = true;
        renderMode();

        expect(await screen.findByText('Autopilot: approach cpu, attempt 1/2')).toBeInTheDocument();
        expect(screen.getByRole('switch', { name: 'Teleoperate' })).toBeInTheDocument();
    });

    it('shows no autopilot state for a leader arm', async () => {
        renderMode();

        expect(await screen.findByRole('button', { name: /Teleoperator/ })).toBeInTheDocument();
        expect(screen.queryByText(/Autopilot:/)).not.toBeInTheDocument();
    });
});

describe("RecordMode relaying the simulation's automatic recording", () => {
    it('runs what the simulation asks for and reports back how it went', async () => {
        session.autopilot = true;
        recording = { phase: 'waiting', message: '', saved: 1, discarded: 0, max_episodes: 0 };
        const bodies: unknown[] = [];
        let replies = 0;
        server.use(
            http.post(
                '/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay',
                async ({ request }) => {
                    bodies.push(await request.json());
                    replies += 1;
                    return HttpResponse.json({
                        active: true,
                        commands:
                            replies === 1
                                ? [{ command: 'start_episode', request_id: 'sim-1', task: 'Put the cpu away' }]
                                : [],
                    });
                }
            )
        );
        renderMode();

        expect(
            await screen.findByText(/Recording from the simulation: waiting \(1 saved, 0 discarded\)/)
        ).toBeInTheDocument();
        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'start_episode', task: 'Put the cpu away' }));
        await waitFor(() =>
            expect(bodies).toContainEqual({
                session: {
                    dataset: false,
                    recording: false,
                    teleop: false,
                    follower: 'Sim arm',
                    leader: 'Sim arm autopilot',
                },
                results: [{ request_id: 'sim-1', ok: true, error: null }],
            })
        );
    });

    it('says so when the simulation cannot be relayed', async () => {
        recording = { phase: 'connecting', message: '', saved: 0, discarded: 0, max_episodes: 0 };
        server.use(
            http.post('/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay', () =>
                HttpResponse.json(
                    {
                        error_code: 'simulation_not_serving',
                        message:
                            'This simulation cannot be recorded automatically from Studio: update physicalai-mujoco-plugin.',
                        http_status: 409,
                    } as never,
                    { status: 409 }
                )
            )
        );
        renderMode();

        expect(await screen.findByText(/update physicalai-mujoco-plugin/)).toBeInTheDocument();
        expect(send).not.toHaveBeenCalled();
    });

    it('relays nothing while the simulation does not record', async () => {
        const relayed = rs.fn();
        server.use(
            http.post('/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay', () => {
                relayed();
                return HttpResponse.json({ active: false, commands: [] });
            })
        );
        renderMode();

        expect(await screen.findByRole('button', { name: /Teleoperator/ })).toBeInTheDocument();
        await new Promise((resolve) => setTimeout(resolve, 600));
        expect(relayed).not.toHaveBeenCalled();
        expect(screen.queryByText(/Recording from the simulation/)).not.toBeInTheDocument();
    });
});

describe('describeAutopilot', () => {
    it('says why the autopilot stalled', () => {
        expect(
            describeAutopilot({ ...autopilot, outcome: 'stalled', failure: 'the arm does not follow the targets' })
        ).toEqual({ variant: 'negative', text: 'stalled: the arm does not follow the targets' });
    });

    it('says why a scene has no autopilot', () => {
        expect(
            describeAutopilot({ ...autopilot, available: false, unavailable_reason: 'No demonstrator for this arm' })
        ).toEqual({ variant: 'negative', text: 'No demonstrator for this arm' });
    });

    it('says the autopilot waits for teleop while it is off', () => {
        expect(describeAutopilot({ ...autopilot, mode: 'off', phase: null, target: null }).variant).toBe('neutral');
    });
});
