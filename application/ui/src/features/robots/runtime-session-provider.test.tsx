// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { Options } from 'react-use-websocket';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { getMockedEnvironment } from '../../test-utils/mocks/mock-environment';
import { render } from '../../test-utils/render';
import { useRobotSimulation } from '../simulation/use-robot-simulation';
import { RuntimeSessionProvider } from './runtime-session-provider';

vi.mock('../../components/websockets/use-websocket-with-response', () => ({ default: vi.fn() }));
vi.mock('../simulation/use-robot-simulation', () => ({ useRobotSimulation: vi.fn() }));

const environment = getMockedEnvironment({
    robots: [
        {
            robot: {
                id: 'follower-1',
                name: 'Sim arm',
                type: 'MuJoCo_SO101_Follower',
                payload: {},
            } as never,
            tele_operator: { type: 'none' } as never,
        },
    ],
});

const lastSocket = () => {
    const calls = vi.mocked(useWebSocketWithResponse).mock.calls;
    const [, options, connect] = calls[calls.length - 1];
    return { options: options as Options, connect };
};

const receive = (message: object) =>
    act(() => lastSocket().options.onMessage?.({ data: JSON.stringify(message) } as MessageEvent));

const probe = (running: boolean | undefined, dataUpdatedAt = 0) =>
    vi.mocked(useRobotSimulation).mockReturnValue({
        data: running === undefined ? undefined : { running },
        dataUpdatedAt,
    } as never);

describe('RuntimeSessionProvider', () => {
    beforeEach(() => {
        vi.mocked(useWebSocketWithResponse).mockReturnValue({
            sendJsonMessage: vi.fn(),
            sendJsonMessageAndWait: vi.fn(),
            readyState: 1,
        } as never);
        probe(undefined);
    });

    const renderProvider = (onError = vi.fn()) => {
        const view = render(
            <RuntimeSessionProvider environment={environment} onError={onError}>
                <div>session</div>
            </RuntimeSessionProvider>,
            { route: '/projects/project-1', path: '/projects/:project_id' }
        );
        return { ...view, onError };
    };

    it('waits for the simulation instead of reporting that it does not run, then connects again', async () => {
        const { onError, findByText } = renderProvider();
        await findByText('session');
        expect(lastSocket().connect).toBe(true);

        await receive({
            event: 'error',
            message: 'No MuJoCo simulation named …',
            error_code: 'simulation_not_running',
        });

        expect(onError).not.toHaveBeenCalled();
        expect(lastSocket().connect).toBe(false);
        expect((lastSocket().options.shouldReconnect as () => boolean)()).toBe(false);
        expect(vi.mocked(useRobotSimulation)).toHaveBeenLastCalledWith('project-1', 'follower-1', true);

        // A probe from before the failure does not count, even if it said running.
        probe(true, 1);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(false);

        probe(true, Date.now() + 1_000);
        await receive({ event: 'state', data: {} });

        expect(lastSocket().connect).toBe(true);
        expect((lastSocket().options.shouldReconnect as () => boolean)()).toBe(true);
    });

    it('reports other errors', async () => {
        const { onError, findByText } = renderProvider();
        await findByText('session');

        await receive({ event: 'error', message: 'Robot unplugged', error_code: 'robot_connection_failed' });

        expect(onError).toHaveBeenCalledWith('Robot unplugged');
        expect(lastSocket().connect).toBe(true);
    });
});
