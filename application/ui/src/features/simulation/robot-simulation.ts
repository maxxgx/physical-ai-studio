// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaRobotSimulationResponse } from '../../api/openapi-spec';
import { useRobotCatalogQuery } from '../robots/robot-catalog.hooks';
import { Simulation } from './use-simulation';

export type RobotSimulationLaunch = SchemaRobotSimulationResponse;

/** How Studio can start the simulation of a robot type, or null when the type is not simulated. */
export const useRobotSimulationLaunch = (robotType: string): RobotSimulationLaunch | null => {
    const { data: catalog } = useRobotCatalogQuery();

    return catalog.find(({ type }) => type === robotType)?.simulation ?? null;
};

/** The robot types Studio can simulate. */
export const useSimulatedRobotTypes = (): Set<string> => {
    const { data: catalog } = useRobotCatalogQuery();

    return new Set(catalog.filter(({ simulation }) => simulation != null).map(({ type }) => type));
};

export type RobotSimulationStatus = 'studio' | 'starting' | 'stopping' | 'external' | 'offline' | 'unknown';

/**
 * Whether a simulated robot's simulation runs, and who started it. Studio's own simulation is
 * matched by robot id; any other simulation that answers under the robot's name (the robot is
 * online) was started outside Studio.
 */
export const robotSimulationStatus = (
    robotId: string,
    simulation: Simulation | undefined,
    connectionStatus: 'online' | 'offline' | 'unknown'
): RobotSimulationStatus => {
    if (simulation?.robot_id === robotId) {
        if (simulation.status === 'running') {
            return 'studio';
        }
        if (simulation.status === 'starting' || simulation.status === 'stopping') {
            return simulation.status;
        }
    }
    if (connectionStatus === 'online') {
        return 'external';
    }

    return connectionStatus;
};

export const ROBOT_SIMULATION_STATUS_LABEL: Record<RobotSimulationStatus, string> = {
    studio: 'Running · Studio',
    starting: 'Starting…',
    stopping: 'Stopping…',
    external: 'Running · external',
    offline: 'Offline',
    unknown: 'Loading...',
};

const SCENE_STORAGE_PREFIX = 'simulation-scene:';

/** The scene last started for a robot, remembered in this browser only. */
export const rememberedScene = (robotId: string, launch: RobotSimulationLaunch): string => {
    try {
        const scene = window.localStorage.getItem(SCENE_STORAGE_PREFIX + robotId);
        if (scene !== null && launch.scenes.some(({ id }) => id === scene)) {
            return scene;
        }
    } catch {
        // Storage can be unavailable (private windows, blocked site data).
    }

    return launch.default_scene;
};

export const rememberScene = (robotId: string, scene: string) => {
    try {
        window.localStorage.setItem(SCENE_STORAGE_PREFIX + robotId, scene);
    } catch {
        // Remembering the scene is a convenience only.
    }
};

export interface StartCommandHint {
    command: string;
    /** The machine to run it on, when the robot attaches to a simulation on another host. */
    host?: string;
}

/** The "start it yourself" command from a simulation-not-running message, if it has one. */
export const startCommandFromMessage = (message: string | null | undefined): StartCommandHint | undefined => {
    const match = message?.match(/Start it(?: on (\S+))? with: (.+)$/);
    if (match === null || match === undefined) {
        return undefined;
    }

    return { command: match[2].trim(), ...(match[1] !== undefined ? { host: match[1] } : {}) };
};
