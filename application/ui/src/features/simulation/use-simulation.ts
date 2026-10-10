// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useQueryClient } from '@tanstack/react-query';

import { $api, fetchClient } from '../../api/client';
import { SchemaSimulation } from '../../api/openapi-spec';

export type Simulation = SchemaSimulation;

/** A simulation Studio started that is still starting, running or stopping. */
export const isActiveSimulation = (simulation: Simulation | undefined): simulation is Simulation =>
    simulation !== undefined &&
    (simulation.status === 'starting' || simulation.status === 'running' || simulation.status === 'stopping');

const simulationQueryKey = () => $api.queryOptions('get', '/api/simulation').queryKey;

/**
 * The simulation Studio started, if any (there is at most one). `useSimulationUpdates` keeps it
 * current; the slow refetch is only a fallback for a dropped event stream.
 */
export const useSimulation = () => {
    return $api.useQuery('get', '/api/simulation', undefined, {
        refetchInterval: (query) => (isActiveSimulation(query.state.data) ? 5_000 : false),
    });
};

/**
 * Owns the one simulation event stream. Mounted once from `AppFooter`: every state change lands
 * in the `useSimulation` query cache, and the browser reconnects the stream on its own (the first
 * event after a reconnect is the full current state).
 */
export const useSimulationUpdates = () => {
    const queryClient = useQueryClient();

    useEffect(() => {
        if (typeof EventSource === 'undefined') {
            return;
        }
        const events = new EventSource(fetchClient.PATH('/api/simulation/events'));

        let lastStatus: Simulation['status'] | undefined;
        events.onmessage = (event: MessageEvent<string>) => {
            let simulation: Simulation;
            try {
                simulation = JSON.parse(event.data) as Simulation;
            } catch {
                // A malformed event is ignored; the next one carries the full state again.
                return;
            }
            queryClient.setQueryData(simulationQueryKey(), simulation);
            if (lastStatus !== undefined && simulation.status !== lastStatus) {
                // Whether a simulated robot is online follows its simulation.
                queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/robots/online'] });
            }
            lastStatus = simulation.status;
        };

        return () => events.close();
    }, [queryClient]);
};

const useSetSimulation = () => {
    const queryClient = useQueryClient();

    return (simulation: Simulation) => queryClient.setQueryData(simulationQueryKey(), simulation);
};

export const useStartSimulation = (project_id: string) => {
    const setSimulation = useSetSimulation();

    return $api.useMutation('post', '/api/simulation', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }]],
        },
        onSuccess: setSimulation,
    });
};

export const useStopSimulation = (project_id?: string) => {
    const setSimulation = useSetSimulation();

    return $api.useMutation('delete', '/api/simulation', {
        meta:
            project_id === undefined
                ? { skipInvalidation: true }
                : {
                      invalidates: [
                          ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
                      ],
                  },
        onSuccess: setSimulation,
    });
};

export const useSimulationLogs = (enabled: boolean) => {
    return $api.useQuery('get', '/api/simulation/logs', undefined, {
        enabled,
        refetchInterval: enabled ? 1_000 : false,
    });
};
