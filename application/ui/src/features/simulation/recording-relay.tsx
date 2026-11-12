// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { StatusLight } from '@geti-ui/ui';

import { fetchClient } from '../../api/client';
import {
    SchemaAckMessage,
    SchemaSessionState,
    SchemaSimulationRecording,
    SchemaSimulationRelayCommand,
    SchemaSimulationRelayResult,
    SchemaSimulationRelaySession,
} from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { useRobotSimulation } from './use-robot-simulation';

/** How often the page relays while the simulation records, so a step waits a fraction of a second at most. */
export const RELAY_INTERVAL_MS = 250;
/** How long the page waits before relaying again after the simulation refused or did not answer. */
const RETRY_MS = 5_000;
/** How long a relayed command may take; the simulation gives up on a save after two minutes. */
const COMMAND_TIMEOUT_MS = 150_000;

const ENDED = new Set(['off', 'done', 'stopped', 'error']);

/** Whether the simulation's automatic recording runs, so that this page relays it. */
export const isRecordingActive = (recording: SchemaSimulationRecording | null | undefined) =>
    recording != null && !ENDED.has(recording.phase);

/** What the page tells the simulation about its session. */
export const relaySession = (
    state: SchemaSessionState | null,
    follower: string | null,
    leader: string | null
): SchemaSimulationRelaySession => ({
    dataset: state?.dataset != null,
    recording: state?.dataset?.is_recording ?? false,
    teleop: state?.control?.type === 'teleop',
    follower,
    leader,
});

type Send = (
    command: { command: 'start_episode'; task: string } | { command: 'save_episode' } | { command: 'discard_episode' }
) => Promise<SchemaAckMessage>;

/** Run one of the simulation's commands as the page's own buttons do, and say how it went. */
export const runRelayed = async (
    command: SchemaSimulationRelayCommand,
    send: Send
): Promise<SchemaSimulationRelayResult> => {
    const request =
        command.command === 'start_episode'
            ? ({ command: 'start_episode', task: command.task ?? '' } as const)
            : ({ command: command.command } as const);
    const timeout = new Promise<SchemaAckMessage>((resolve) =>
        window.setTimeout(
            () =>
                resolve({
                    type: 'ack',
                    request_id: '',
                    ok: false,
                    error: { message: 'Studio did not answer in time.', error_code: 'timeout' },
                }),
            COMMAND_TIMEOUT_MS
        )
    );
    const ack = await Promise.race([send(request), timeout]);
    return { request_id: command.request_id, ok: ack.ok, error: ack.ok ? null : (ack.error?.message ?? 'failed') };
};

/**
 * Relay the automatic recording of the robot's simulation ("Record in Studio" in its viewer) while it runs.
 *
 * The simulation decides when an episode starts and whether it is kept; this page tells it about
 * the session and runs what it asks for, through the session's own commands. Shows how it goes.
 */
export const SimulationRecordingRelay = ({
    robot_id,
    state,
    follower,
    leader,
    send,
}: {
    robot_id: string;
    state: SchemaSessionState | null;
    follower: string | null;
    leader: string | null;
    send: Send;
}) => {
    const { project_id } = useProjectId();
    const { data: simulation } = useRobotSimulation(project_id, robot_id);
    const recording = simulation?.running === true ? simulation.recording : null;
    const active = isRecordingActive(recording);
    const [failure, setFailure] = useState<string | null>(null);

    // Read by the relay loop, which must not restart on every state message.
    const session = useRef(relaySession(state, follower, leader));
    session.current = relaySession(state, follower, leader);
    const sendRef = useRef(send);
    sendRef.current = send;

    useEffect(() => {
        if (!active) {
            return;
        }
        let stopped = false;
        let busy = false;
        let retryAt = 0;
        let results: SchemaSimulationRelayResult[] = [];
        const exchange = async (last = false) => {
            if (busy || (stopped && !last) || (!last && Date.now() < retryAt)) {
                return;
            }
            busy = true;
            try {
                const sending = results;
                results = [];
                const { data, error } = await fetchClient.POST(
                    '/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay',
                    {
                        params: { path: { project_id, robot_id } },
                        body: { session: session.current, results: sending },
                    }
                );
                if (data === undefined) {
                    results = [...sending, ...results]; // not delivered: send them again
                    retryAt = Date.now() + RETRY_MS;
                    const message = (error as { message?: unknown } | undefined)?.message;
                    setFailure(typeof message === 'string' ? message : 'The simulation did not answer.');
                    return;
                }
                setFailure(null);
                for (const command of data.commands ?? []) {
                    results.push(await runRelayed(command, sendRef.current));
                }
            } finally {
                busy = false;
            }
        };
        const timer = window.setInterval(() => void exchange(), RELAY_INTERVAL_MS);
        return () => {
            stopped = true;
            window.clearInterval(timer);
            // Once more: a run that just ended can still have a last discard, and wants the latest results.
            void exchange(true);
        };
    }, [active, project_id, robot_id]);

    if (recording == null || recording.phase === 'off') {
        return null;
    }
    const counts = `${recording.saved} saved, ${recording.discarded} discarded`;
    const variant = recording.phase === 'error' || (active && failure) ? 'negative' : active ? 'info' : 'neutral';
    const message = active && failure ? failure : recording.message || recording.phase;
    return (
        <StatusLight variant={variant}>
            Recording from the simulation: {message} ({counts})
        </StatusLight>
    );
};
