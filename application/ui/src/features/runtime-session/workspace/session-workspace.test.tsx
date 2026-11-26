// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSyncExternalStore } from 'react';

import { toast } from '@geti-ui/ui';
import { afterEach, beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { SessionWorkspace } from './session-workspace';

// model-c was trained on another dataset with the same default task as model-a; model-m only
// takes a fixed set of tasks, which its dataset's default is not one of.
const DEFAULT_TASKS: Record<string, string> = {
    'model-a': 'Pick the cube',
    'model-b': 'Sort the bucket',
    'model-c': 'Pick the cube',
    'model-m': 'Do something else',
};
const FIXED_TASKS: Record<string, string[]> = { 'model-m': ['pick', 'place'] };
const TYPED = 'Put the red cube in the bucket';

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
                model: {
                    model_id,
                    name: model_id,
                    backend: 'torch',
                    device: 'cpu',
                    tasks: FIXED_TASKS[model_id] ?? [],
                },
            },
            statuses: {},
            send,
        };
    },
}));

/** The datasets whose default task the backend has served, in order. */
let served: string[] = [];
/** How the next request for a dataset's default task goes: fails, waits for a release, or is served. */
let nextDataset: 'fail' | Promise<void> | null = null;

const mockModels = () =>
    server.use(
        http.get(
            '/api/models/{model_id}',
            ({ params }) =>
                HttpResponse.json({
                    model: { id: params.model_id, dataset_id: `dataset-of-${params.model_id}` },
                }) as never
        ),
        http.get('/api/dataset/{dataset_id}', async ({ params }) => {
            const dataset_id = String(params.dataset_id);
            const next = nextDataset;
            nextDataset = null;
            if (next === 'fail') {
                return HttpResponse.json({ message: 'unavailable' }, { status: 503 }) as never;
            }
            await next;
            served.push(dataset_id);
            return HttpResponse.json({
                id: dataset_id,
                default_task: DEFAULT_TASKS[dataset_id.replace('dataset-of-', '')],
            }) as never;
        })
    );

const renderWorkspace = (project_id = 'project-1') =>
    render(<SessionWorkspace initialMode='inference' cancelHref='/' />, {
        route: `/projects/${project_id}/inference`,
        path: '/projects/:project_id/inference',
    });

type Scope = Pick<typeof screen, 'findByRole'>;

const taskBox = (scope: Scope = screen) => scope.findByRole('textbox', { name: 'Task' });

const pick = async (label: RegExp, option: string) => {
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: label }));
    await user.click(await screen.findByRole('option', { name: option }));
};

const switchMode = async (option: 'Record' | 'Inference') => {
    await pick(/Mode/, option);
};

const setSlider = async (name: string, value: number, scope: Scope = screen) => {
    fireEvent.change(await scope.findByRole('slider', { name }), { target: { value: String(value) } });
};

/** Load ``id`` and wait until its default task has been served and rendered. */
const loadModel = async (id: string) => {
    const before = served.length;
    act(() => swapModel(id));
    // A default served earlier comes from the cache; wait for the new one only.
    if (!served.includes(`dataset-of-${id}`)) {
        await waitFor(() => expect(served.slice(before)).toContain(`dataset-of-${id}`));
    }
    await act(async () => {});
};

/** The model control the next Play sends. */
const played = async () => {
    send.mockClear();
    fireEvent.click(await screen.findByRole('button', { name: 'Play' }));
    await waitFor(() => expect(send).toHaveBeenCalled());
    const [command] = send.mock.calls.at(-1) as unknown as [{ control: unknown }];
    return command.control;
};

/** Store ``values`` as a tab of project-1 would have, each setting under its own key. */
const storeRaw = (values: Record<string, string>) =>
    Object.entries(values).forEach(([field, raw]) =>
        window.localStorage.setItem(`physicalai:inferenceSettings:project-1:${field}`, raw)
    );

beforeEach(() => {
    window.localStorage.clear();
    modelId = 'model-a';
    served = [];
    nextDataset = null;
    mockModels();
});

afterEach(() => {
    send.mockClear();
});

describe('SessionWorkspace inference settings', () => {
    it('keeps the task and every execution setting across a switch to Record and back, and a reload', async () => {
        const first = renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        fireEvent.change(await taskBox(), { target: { value: TYPED } });
        await pick(/Execution/, 'Asynchronous');
        await setSlider('Action rate', 15);
        await setSlider('Request the next chunk at', 0.2);

        await switchMode('Record');
        expect(await screen.findByText('Record panel')).toBeInTheDocument();
        await switchMode('Inference');

        const expected = { task: TYPED, hz: 15, execution: { mode: 'async', request_threshold: 0.2 } };
        expect(await taskBox()).toHaveValue(TYPED);
        expect(await played()).toMatchObject(expected);

        first.unmount();
        renderWorkspace();
        expect(await taskBox()).toHaveValue(TYPED);
        expect(await played()).toMatchObject(expected);
    });

    it('keeps a typed task when another model is loaded', async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        fireEvent.change(await taskBox(), { target: { value: TYPED } });
        await loadModel('model-b');

        expect(await taskBox()).toHaveValue(TYPED);
    });

    it('keeps a model default that was run without editing when another model is loaded', async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        await loadModel('model-b');

        expect(await taskBox()).toHaveValue('Pick the cube');
        expect(await played()).toMatchObject({ task: 'Pick the cube' });
    });

    it("fills a cleared task with the next model's default, also when it is the same text", async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        // Clear it with model-a's default cached, then load model-c, whose default is the same text.
        await loadModel('model-c');
        await loadModel('model-a');
        fireEvent.change(await taskBox(), { target: { value: '' } });
        await loadModel('model-c');

        expect(await taskBox()).toHaveValue('Pick the cube');
    });

    it('keeps a cleared task cleared on a reload with the same model', async () => {
        const first = renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        fireEvent.change(await taskBox(), { target: { value: '' } });
        first.unmount();

        served = [];
        renderWorkspace();
        await waitFor(() => expect(served).toContain('dataset-of-model-a'));
        await act(async () => {});

        expect(await taskBox()).toHaveValue('');
    });

    it('keeps what each of two tabs on one project changed, not the last tab’s whole draft', async () => {
        const tabA = renderWorkspace();
        const tabB = renderWorkspace();
        const a = within(tabA.container);
        const b = within(tabB.container);
        await waitFor(async () => expect(await taskBox(a)).toHaveValue('Pick the cube'));
        await waitFor(async () => expect(await taskBox(b)).toHaveValue('Pick the cube'));

        await setSlider('Request the next chunk at', 0.2, a);
        fireEvent.change(await taskBox(b), { target: { value: TYPED } });
        await setSlider('Action rate', 15, b);
        tabA.unmount();
        tabB.unmount();

        renderWorkspace();
        expect(await taskBox()).toHaveValue(TYPED);
        expect(await played()).toMatchObject({ task: TYPED, hz: 15, execution: { request_threshold: 0.2 } });
    });

    it.each([
        { name: 'out of range', stored: { task: 5, mode: 'rtc', hz: 250, threshold: 1.5 } },
        { name: 'off the steps', stored: { mode: 'async', hz: 7, threshold: 0.33 } },
        { name: 'of the wrong type', stored: { task: ['x'], mode: 3, hz: '15', threshold: null } },
    ])('runs the defaults in place of stored values $name', async ({ stored }) => {
        storeRaw(Object.fromEntries(Object.entries(stored).map(([field, value]) => [field, JSON.stringify(value)])));
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        expect(await played()).toMatchObject({
            task: 'Pick the cube',
            hz: 30,
            execution: { mode: stored.mode === 'async' ? 'async' : 'sync', request_threshold: 0.5 },
        });
    });

    it('starts from the defaults when the stored settings are not JSON', async () => {
        storeRaw({ task: '{not json', mode: 'async', hz: '1e', threshold: '' });
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));

        expect(await played()).toMatchObject({
            task: 'Pick the cube',
            hz: 30,
            execution: { mode: 'sync', request_threshold: 0.5 },
        });
    });

    it('keeps the settings of each project apart', async () => {
        const first = renderWorkspace('project-1');
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        fireEvent.change(await taskBox(), { target: { value: TYPED } });
        await pick(/Execution/, 'Asynchronous');
        first.unmount();

        renderWorkspace('project-2');
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        expect(await played()).toMatchObject({ task: 'Pick the cube', execution: { mode: 'sync' } });
    });

    it('keeps the task another tab typed while this tab still waited for the default', async () => {
        let release = () => {};
        nextDataset = new Promise<void>((resolve) => (release = resolve));
        const waiting = within(renderWorkspace().container);
        // The other tab's default arrives at once; it is filled there, then replaced by a typed task.
        const typing = renderWorkspace();
        const b = within(typing.container);
        await waitFor(async () => expect(await taskBox(b)).toHaveValue('Pick the cube'));
        fireEvent.change(await taskBox(b), { target: { value: TYPED } });
        typing.unmount();

        await act(async () => release());
        await waitFor(() => expect(served.filter((id) => id === 'dataset-of-model-a')).toHaveLength(2));
        await act(async () => {});
        expect(await taskBox(waiting)).toHaveValue(TYPED);
        expect(window.localStorage.getItem('physicalai:inferenceSettings:project-1:task')).toBe(JSON.stringify(TYPED));
    });

    it('keeps the task a fixed-task model ran when a free-text model is loaded next', async () => {
        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
        fireEvent.change(await taskBox(), { target: { value: TYPED } });

        await loadModel('model-m');
        expect(await played()).toMatchObject({ task: 'pick' });
        await loadModel('model-b');

        expect(await taskBox()).toHaveValue('pick');
    });

    it("fills the task once a model's default task can be fetched after a failed fetch", async () => {
        nextDataset = 'fail';
        const first = renderWorkspace();
        await waitFor(async () => expect(await screen.findByRole('textbox', { name: 'Task' })).toBeInTheDocument());
        await act(async () => {});
        expect(await taskBox()).toHaveValue('');
        first.unmount();

        renderWorkspace();
        await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
    });

    it('says once that the settings will not be kept when the browser blocks storage', async () => {
        const blocked = () => {
            throw new DOMException('blocked', 'SecurityError');
        };
        const getItem = rs.spyOn(Storage.prototype, 'getItem').mockImplementation(blocked);
        const setItem = rs.spyOn(Storage.prototype, 'setItem').mockImplementation(blocked);
        const notify = rs.spyOn(toast, 'info').mockReturnValue(rs.fn() as never);
        try {
            renderWorkspace();
            await waitFor(async () => expect(await taskBox()).toHaveValue('Pick the cube'));
            fireEvent.change(await taskBox(), { target: { value: TYPED } });
            await setSlider('Action rate', 15);
            await switchMode('Record');
            await switchMode('Inference');

            // The settings still hold on this page.
            expect(await played()).toMatchObject({ task: TYPED, hz: 15 });
            expect(notify).toHaveBeenCalledTimes(1);
            expect(String(notify.mock.calls[0][0])).toMatch(/not be kept/);
        } finally {
            getItem.mockRestore();
            setItem.mockRestore();
            notify.mockRestore();
        }
    });
});
