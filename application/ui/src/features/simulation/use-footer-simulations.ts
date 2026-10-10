// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { useOptionalProjectId } from '../projects/use-project';
import { isActiveSimulation, useSimulation } from './use-simulation';

export interface FooterSimulation {
    key: string;
    label: string;
    scene: string | null;
    startedByStudio: boolean;
    httpUrl: string | null;
    viewerUrl: string | null;
    realTimeFactor: number | null;
}

const port = (url: string | null | undefined) => {
    if (url == null) {
        return undefined;
    }
    try {
        return new URL(url).port || undefined;
    } catch {
        return undefined;
    }
};

export const simulationPort = port;

/**
 * The simulations to list in the footer: Studio's own (from any project) and those that run for
 * the current project's simulated robots, also when started elsewhere (CLI).
 */
export const useFooterSimulations = (): FooterSimulation[] => {
    const { project_id } = useOptionalProjectId();
    const { data: studio } = useSimulation();
    const { data: project } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/simulations',
        { params: { path: { project_id: project_id ?? '' } } },
        { enabled: project_id !== undefined, refetchInterval: 5_000 }
    );

    const rows: FooterSimulation[] = (project_id === undefined ? [] : (project ?? [])).map((simulation) => ({
        key: simulation.robot_id,
        label: simulation.robot_name,
        scene: simulation.scene ?? null,
        startedByStudio: simulation.started_by === 'studio',
        httpUrl: simulation.http_url ?? null,
        viewerUrl: simulation.viewer_url ?? null,
        realTimeFactor: simulation.real_time_factor ?? null,
    }));

    if (isActiveSimulation(studio) && !rows.some(({ key }) => key === studio.robot_id)) {
        rows.unshift({
            key: studio.robot_id ?? 'studio',
            label: (studio.robot_type ?? 'Simulation').replaceAll('_', ' '),
            scene: studio.scene ?? null,
            startedByStudio: true,
            httpUrl: studio.http_url ?? null,
            viewerUrl: studio.viewer_url ?? null,
            realTimeFactor: null,
        });
    }

    return rows;
};
