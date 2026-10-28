// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';

import { $api } from '../../api/client';
import { createQueryClient } from '../../query-client/query-client';
import { Simulation, useSimulationUpdates } from './use-simulation';

class FakeEventSource {
    static readonly CLOSED = 2;
    static instances: FakeEventSource[] = [];
    readyState = 1;
    onmessage: ((event: MessageEvent<string>) => void) | null = null;
    onerror: (() => void) | null = null;
    closed = false;

    constructor(readonly url: string) {
        FakeEventSource.instances.push(this);
    }

    emit(simulation: Partial<Simulation>) {
        this.onmessage?.(new MessageEvent('message', { data: JSON.stringify({ cameras: [], ...simulation }) }));
    }

    close() {
        this.closed = true;
    }
}

const renderUpdates = () => {
    const queryClient = createQueryClient();
    const view = renderHook(() => useSimulationUpdates(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
    });

    const events = () => {
        const latest = FakeEventSource.instances.at(-1);
        if (latest === undefined) {
            throw new Error('No event stream was opened');
        }
        return latest;
    };

    return { queryClient, view, events };
};

describe('useSimulationUpdates', () => {
    afterEach(() => {
        FakeEventSource.instances = [];
        rs.unstubAllGlobals();
        rs.useRealTimers();
    });

    it('writes every event into the simulation query', () => {
        rs.stubGlobal('EventSource', FakeEventSource);
        const { queryClient, events } = renderUpdates();

        act(() => events().emit({ status: 'running', robot_id: 'robot-1' }));

        expect(queryClient.getQueryData($api.queryOptions('get', '/api/simulation').queryKey)).toMatchObject({
            status: 'running',
            robot_id: 'robot-1',
        });
        expect(events().url).toContain('/api/simulation/events');
    });

    it('refreshes the robots online status when the simulation status changes', () => {
        rs.stubGlobal('EventSource', FakeEventSource);
        const { queryClient, events } = renderUpdates();
        const invalidate = rs.spyOn(queryClient, 'invalidateQueries');

        act(() => events().emit({ status: 'starting' }));
        act(() => events().emit({ status: 'starting', phase: 'load' }));
        expect(invalidate).not.toHaveBeenCalled();

        act(() => events().emit({ status: 'running' }));
        expect(invalidate).toHaveBeenCalledWith({ queryKey: ['get', '/api/projects/{project_id}/robots/online'] });
    });

    it('refreshes the robots once a started simulation is running, as it stored its new address on its robot', () => {
        rs.stubGlobal('EventSource', FakeEventSource);
        const { queryClient, events } = renderUpdates();
        const path = { params: { path: { project_id: 'project-1', robot_id: 'robot-1' } } };
        const robotKey = $api.queryOptions('get', '/api/projects/{project_id}/robots/{robot_id}', path).queryKey;
        const robotsKey = $api.queryOptions('get', '/api/projects/{project_id}/robots', path).queryKey;
        queryClient.setQueryData(robotKey, { payload: { http_url: 'http://127.0.0.1:8080' } });
        queryClient.setQueryData(robotsKey, []);

        act(() => events().emit({ status: 'starting', robot_id: 'robot-1' }));
        act(() => events().emit({ status: 'running', robot_id: 'robot-1', http_url: 'http://127.0.0.1:56209' }));

        expect(queryClient.getQueryState(robotKey)?.isInvalidated).toBe(true);
        expect(queryClient.getQueryState(robotsKey)?.isInvalidated).toBe(true);
    });

    it('opens a new stream after the browser gave up on one, and closes it on unmount', () => {
        rs.useFakeTimers();
        rs.stubGlobal('EventSource', FakeEventSource);
        const { view, events } = renderUpdates();
        const first = events();

        first.readyState = FakeEventSource.CLOSED;
        act(() => first.onerror?.());
        act(() => rs.advanceTimersByTime(3_000));

        expect(FakeEventSource.instances).toHaveLength(2);
        view.unmount();
        expect(events().closed).toBe(true);
    });
});
