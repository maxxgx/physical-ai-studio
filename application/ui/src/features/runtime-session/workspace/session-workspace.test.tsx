// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSyncExternalStore } from 'react';

import { afterEach, beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { SessionWorkspace } from './session-workspace';

const DEFAULT_TASKS: Record<string, string> = { 'model-a': 'Pick the cube', 'model-b': 'Sort the bucket' };

const send = rs.fn(async () => ({ ok: true }));

// The loaded model, swappable while the workspace stays mounted.
let modelId = 'model-a';
const listeners = new Set<() => void>();
const swapModel = (id: string) => {
    modelId = id;
    listeners.forEach((listener) => listener());
};
const useModelId = () =>
    useSyncExternalStore(
        (listener) => {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        () => modelId
    );

rs.mock('../session-viewer', () => ({ SessionViewer: () => <div>Viewer</div> }));
rs.mock('./parts-panel', () => ({ PartsPanel: () => null }));
rs.mock('./record-mode', () => ({ RecordMode: () => <div>Record panel</div> }));

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    useRuntimeSession: () => {
        const model_id = useModelId();
        return {
            state: {
                environment: { name: 'Bench', cameras: [] },
                dataset: { is_recording: false },
                model: { model_id, name: model_id, backend: 'torch', device: 'cpu', tasks: [] },
            },
            statuses: {},
            send,
        };
    },
}));

const mockModels = () =>
    server.use(
        http.get(
            '/api/models/{model_id}',
            ({ params }) =>
                HttpResponse.json({
                    model: { id: params.model_id, dataset_id: `dataset-of-${params.model_id}` },
                }) as never
        ),
        http.get(
            '/api/dataset/{dataset_id}',
            ({ params }) =>
                HttpResponse.json({
                    id: params.dataset_id,
                    default_task: DEFAULT_TASKS[String(params.dataset_id).replace('dataset-of-', '')],
                }) as never
        )
    );

const renderWorkspace = (project_id = 'project-1') =>
    render(<SessionWorkspace initialMode='inference' cancelHref='/' />, {
        route: `/projects/${project_id}/inference`,
        path: '/projects/:project_id/inference',
    });

const taskBox = () => screen.findByRole('textbox', { name: 'Task' });

const pick = async (label: RegExp, option: string) => {
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: label }));
    await user.click(await screen.findByRole('option', { name: option }));
};

const switchMode = async (option: 'Record' | 'Inference') => {
    await pick(/Mode/, option);
};

/** The model control the next Play sends. */
const played = async () => {
    send.mockClear();
    fireEvent.click(await screen.findByRole('button', { name: 'Play' }));
    await waitFor(() => expect(send).toHaveBeenCalled());
    const [command] = send.mock.calls.at(-1) as unknown as [{ control: unknown }];
    return command.control;
};

beforeEach(() => {
    window.localStorage.clear();
    modelId = 'model-a';
    mockModels();
});

afterEach(() => {
    send.mockClear();
});

describe('SessionWorkspace inference settings', () => {
    it('keeps a typed task and the execution settings across a switch to Record and back', async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        fireEvent.change(await taskBox(), { target: { value: 'Put the red cube in the bucket' } });
        await pick(/Execution/, 'Asynchronous');

        await switchMode('Record');
        expect(await screen.findByText('Record panel')).toBeInTheDocument();
        await switchMode('Inference');

        expect(await taskBox()).toHaveValue('Put the red cube in the bucket');
        expect(await played()).toMatchObject({
            task: 'Put the red cube in the bucket',
            execution: { mode: 'async' },
        });
    });

    it('keeps a typed task when another model is loaded, and fills an empty one with its default', async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        fireEvent.change(await taskBox(), { target: { value: 'Put the red cube in the bucket' } });
        act(() => swapModel('model-b'));
        // Give the new model's default task time to arrive; it must not replace the typed one.
        await screen.findByText('Viewer');
        await new Promise((resolve) => setTimeout(resolve, 100));
        expect(await taskBox()).toHaveValue('Put the red cube in the bucket');

        fireEvent.change(await taskBox(), { target: { value: '' } });
        act(() => swapModel('model-a'));
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
    });

    it('restores the task and settings on the next page load, for the same project only', async () => {
        const first = renderWorkspace('project-1');
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        fireEvent.change(await taskBox(), { target: { value: 'Put the red cube in the bucket' } });
        await pick(/Execution/, 'Asynchronous');
        first.unmount();

        const second = renderWorkspace('project-1');
        expect(await taskBox()).toHaveValue('Put the red cube in the bucket');
        expect(await played()).toMatchObject({ execution: { mode: 'async' } });
        second.unmount();

        renderWorkspace('project-2');
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        const execution = await screen.findByRole('button', { name: /Execution/ });
        expect(within(execution).getByText('Synchronous')).toBeInTheDocument();
    });
});
