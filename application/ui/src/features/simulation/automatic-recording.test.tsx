// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { AutomaticRecording } from './automatic-recording';

const ROBOT_ID = 'robot-1';
const send = rs.fn(async (_command: unknown) => ({ type: 'ack' as const, request_id: '', ok: true }));
let recording: Record<string, unknown> | null = null;
let posted: unknown[] = [];

const RECORDING = {
    available: true,
    task_source: 'instruction',
    phase: 'off',
    message: '',
    saved: 0,
    discarded: 0,
    max_episodes: 0,
    keep: 'perfect',
};

beforeEach(() => {
    send.mockClear();
    posted = [];
    recording = { ...RECORDING };
    server.use(
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({
                running: true,
                started_by: 'studio',
                owner_name: 'arm',
                cameras: [],
                recording,
            } as never)
        ),
        http.post('/api/projects/{project_id}/robots/{robot_id}/simulation/recording', async ({ request }) => {
            posted.push(await request.json());
            return new HttpResponse(null, { status: 204 }) as never;
        }),
        http.post('/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay', () =>
            HttpResponse.json({ active: false, owner: false, commands: [] })
        )
    );
});

const renderPanel = (state: Record<string, unknown> = { session_id: 's1', control: null, dataset: null }) =>
    render(
        <AutomaticRecording
            robot_id={ROBOT_ID}
            state={state as never}
            follower='Sim arm'
            leader='Sim arm autopilot'
            defaultTask='Sort the parts'
            canRecord
            send={send as never}
        />,
        { route: '/projects/project-1/record', path: '/projects/:project_id/record' }
    );

describe('AutomaticRecording', () => {
    it('starts teleoperation and the simulation recording with the settings', async () => {
        renderPanel();
        const user = userEvent.setup();

        const button = await screen.findByRole('button', { name: 'Record automatically' });
        await waitFor(() => expect(button).toBeEnabled());
        expect(screen.queryByRole('textbox', { name: /Task/ })).not.toBeInTheDocument(); // each instruction is the task
        await user.clear(screen.getByRole('textbox', { name: 'Episodes' }));
        await user.type(screen.getByRole('textbox', { name: 'Episodes' }), '3');
        await user.click(screen.getByRole('radio', { name: 'All' }));
        await user.click(button);

        await waitFor(() =>
            expect(posted).toEqual([
                { enabled: true, keep: 'all', max_episodes: 3, recovery_fraction: 0, max_attempts: 2 },
            ])
        );
        expect(send).toHaveBeenCalledWith({
            command: 'set_control',
            control: { type: 'teleop', hz: 100, grace_s: 2 },
        });
    });

    it('asks for the task of a scene where every episode has the same one', async () => {
        recording = { ...RECORDING, task_source: 'task' };
        renderPanel();
        const user = userEvent.setup();

        const task = await screen.findByRole('textbox', { name: /Task/ });
        expect(task).toHaveValue('Sort the parts');
        expect(screen.queryByText('More')).not.toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Record automatically' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Record automatically' }));

        await waitFor(() => expect(posted).toHaveLength(1));
        expect(posted[0]).toMatchObject({ enabled: true, task: 'Sort the parts' });
    });

    it('shows how a run goes and stops it, with teleoperation', async () => {
        recording = { ...RECORDING, phase: 'recording', message: 'Recording episode 2.', saved: 1 };
        renderPanel({ session_id: 's1', control: { type: 'teleop' }, dataset: null });
        const user = userEvent.setup();

        expect(await screen.findByText(/Recording episode 2\. \(1 saved, 0 discarded\)/)).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Stop recording' }));

        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'set_control', control: null }));
        expect(posted).toEqual([
            { enabled: false, keep: 'perfect', max_episodes: 0, recovery_fraction: 0, max_attempts: 2 },
        ]);
    });

    it('offers nothing in a scene that cannot record automatically', async () => {
        recording = { ...RECORDING, available: false };
        renderPanel();

        await new Promise((resolve) => setTimeout(resolve, 300));
        expect(screen.queryByRole('button', { name: 'Record automatically' })).not.toBeInTheDocument();
    });
});
