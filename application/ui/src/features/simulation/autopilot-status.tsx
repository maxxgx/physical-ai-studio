// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StatusLight } from '@geti-ui/ui';

import { SchemaSimulationAutopilot } from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { useRobotSimulation } from './use-robot-simulation';

type Variant = 'positive' | 'negative' | 'notice' | 'info' | 'neutral';

/** What the simulation's autopilot does, in a few words: "approach cpu, attempt 1/2", or why it does not lead. */
export const describeAutopilot = (
    autopilot: SchemaSimulationAutopilot | null | undefined
): { variant: Variant; text: string } => {
    if (autopilot == null) {
        return { variant: 'neutral', text: 'not reported by this simulation' };
    }
    if (!autopilot.available) {
        return { variant: 'negative', text: autopilot.unavailable_reason ?? 'this scene has none' };
    }
    if (autopilot.outcome === 'stalled' || autopilot.outcome === 'failed') {
        return {
            variant: 'negative',
            text: autopilot.failure ? `${autopilot.outcome}: ${autopilot.failure}` : autopilot.outcome,
        };
    }
    if (autopilot.mode === 'off') {
        return { variant: 'neutral', text: 'off, it leads while you teleoperate' };
    }
    if (autopilot.mode === 'drive') {
        return { variant: 'notice', text: 'driving the arm itself' };
    }
    const doing = [autopilot.phase, autopilot.target].filter(Boolean).join(' ') || 'leading';
    const attempt =
        autopilot.attempt && autopilot.max_attempts ? `, attempt ${autopilot.attempt}/${autopilot.max_attempts}` : '';
    return { variant: autopilot.outcome === 'success' ? 'positive' : 'info', text: `${doing}${attempt}` };
};

/** The autopilot of a simulated robot's simulation, as it last reported; nothing while the simulation does not run. */
export const AutopilotStatus = ({ robot_id }: { robot_id: string }) => {
    const { project_id } = useProjectId();
    const { data: simulation } = useRobotSimulation(project_id, robot_id);

    if (simulation?.running !== true) {
        return null;
    }
    const { variant, text } = describeAutopilot(simulation.autopilot);

    return <StatusLight variant={variant}>Autopilot: {text}</StatusLight>;
};
