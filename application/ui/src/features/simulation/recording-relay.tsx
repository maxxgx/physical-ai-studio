// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { Flex, StatusLight, Text } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { fetchClient } from '../../api/client';
import {
    SchemaAckMessage,
    SchemaSessionState,
    SchemaSimulationRecording,
    SchemaSimulationRelayCommand,
    SchemaSimulationRelayReply,
    SchemaSimulationRelayRequest,
    SchemaSimulationRelayResult,
    SchemaSimulationRelaySession,
} from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { useRobotSimulation } from './use-robot-simulation';

/** How often the page relays while the simulation records, so a step waits a fraction of a second at most. */
export const RELAY_INTERVAL_MS = 250;
/** How long the page waits before relaying again after the simulation refused or did not answer. */
export const RETRY_MS = 5_000;
/** How long the page keeps trying to reach a simulation whose run ended, for its last discard. */
export const GIVE_UP_MS = 15_000;
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
    session_id: state?.session_id ?? null,
    dataset_id: state?.dataset?.dataset_id ?? null,
    dataset: state?.dataset != null,
    recording: state?.dataset?.is_recording ?? false,
    teleop: state?.control?.type === 'teleop',
    follower,
    leader,
});

export type Send = (
    command:
        | { command: 'start_episode'; task: string; dataset_id: string | null; episode_id: string | null }
        | { command: 'save_episode'; dataset_id: string | null; episode_id: string | null }
        | { command: 'discard_episode'; dataset_id: string | null; episode_id: string | null }
) => Promise<SchemaAckMessage>;

/**
 * Run one of the simulation's commands as the page's own buttons do, and say how it went.
 *
 * It names the dataset and episode it is for: the session refuses it if another dataset was loaded
 * or another episode started meanwhile, even if this page has not heard of that yet.
 */
export const runRelayed = async (
    command: SchemaSimulationRelayCommand,
    dataset_id: string | null,
    send: Send
): Promise<SchemaSimulationRelayResult> => {
    // The simulation names the dataset and episode; the page's own dataset is the fallback.
    const fence = { dataset_id: command.dataset_id ?? dataset_id, episode_id: command.episode_id ?? null };
    const request =
        command.command === 'start_episode'
            ? ({ command: 'start_episode', task: command.task ?? '', ...fence } as const)
            : ({ command: command.command, ...fence } as const);
    let timer = 0;
    const timeout = new Promise<SchemaAckMessage>((resolve) => {
        timer = window.setTimeout(
            () =>
                resolve({
                    type: 'ack',
                    request_id: '',
                    ok: false,
                    error: { message: 'Studio did not answer in time.', error_code: 'timeout' },
                }),
            COMMAND_TIMEOUT_MS
        );
    });
    try {
        const ack = await Promise.race([send(request), timeout]);
        return { request_id: command.request_id, ok: ack.ok, error: ack.ok ? null : (ack.error?.message ?? 'failed') };
    } catch (error) {
        return { request_id: command.request_id, ok: false, error: String(error) };
    } finally {
        window.clearTimeout(timer);
    }
};

/** Where the page is: commands received on one session and dataset do not run on another. */
const contextOf = (session: SchemaSimulationRelaySession) => `${session.session_id}/${session.dataset_id}`;

/**
 * The relay of one simulation's run: exchanges every tick, and runs the commands it gets one at a
 * time, apart from the exchanges, so a slow save does not stop them (the simulation counts silence
 * as a closed page).
 *
 * A command comes with every exchange until its result does: each request id runs once, and its
 * result goes with every exchange until the command no longer comes. A command runs only on the
 * session and dataset it came on.
 */
export class RecordingRelayLoop {
    /** Whether the last reply asked for nothing more: no run, no command waiting, none running. */
    idle = false;
    /** Whether this page relays the run; another page may. */
    owner = true;

    private readonly results = new Map<string, SchemaSimulationRelayResult>();
    private readonly running = new Set<string>();
    private queue: { command: SchemaSimulationRelayCommand; context: string; dataset_id: string | null }[] = [];
    private failingSince: number | null = null;
    private exchanging = false;
    private working = false;
    private stopped = false;
    private retryAt = 0;

    constructor(
        private readonly client: string,
        private readonly exchange: (request: SchemaSimulationRelayRequest) => Promise<SchemaSimulationRelayReply>,
        private readonly run: (
            command: SchemaSimulationRelayCommand,
            dataset_id: string | null
        ) => Promise<SchemaSimulationRelayResult>,
        private readonly session: () => SchemaSimulationRelaySession,
        private readonly onFailure: (message: string | null) => void,
        private readonly now: () => number = Date.now
    ) {}

    async tick(): Promise<void> {
        if (this.stopped || this.exchanging || this.now() < this.retryAt) {
            return;
        }
        this.exchanging = true;
        const session = this.session();
        try {
            const reply = await this.exchange({ client: this.client, session, results: [...this.results.values()] });
            if (this.stopped) {
                return;
            }
            this.onFailure(null);
            this.failingSince = null;
            this.owner = reply.owner ?? false;
            const commands = reply.commands ?? [];
            const listed = new Set(commands.map(({ request_id }) => request_id));
            for (const request_id of [...this.results.keys()]) {
                if (!listed.has(request_id)) {
                    this.results.delete(request_id); // the simulation has it
                }
            }
            // One the simulation no longer asks for (its run was switched off) is not run, if not yet started.
            const withdrawn = this.queue.filter(({ command }) => !listed.has(command.request_id));
            this.queue = this.queue.filter(({ command }) => listed.has(command.request_id));
            withdrawn.forEach(({ command }) => this.running.delete(command.request_id));
            for (const command of commands) {
                if (!this.results.has(command.request_id) && !this.running.has(command.request_id)) {
                    this.running.add(command.request_id);
                    this.queue.push({ command, context: contextOf(session), dataset_id: session.dataset_id ?? null });
                }
            }
            this.idle = !reply.active && commands.length === 0 && this.running.size === 0;
            void this.work();
        } catch (error) {
            this.retryAt = this.now() + RETRY_MS;
            this.failingSince ??= this.now();
            this.onFailure(error instanceof Error ? error.message : 'The simulation did not answer.');
        } finally {
            this.exchanging = false;
        }
    }

    stop(): void {
        this.stopped = true;
    }

    /** How long exchanges have been failing, in ms; 0 while they go through. */
    failingFor(): number {
        return this.failingSince === null ? 0 : this.now() - this.failingSince;
    }

    private async work(): Promise<void> {
        if (this.working) {
            return;
        }
        this.working = true;
        try {
            for (let next = this.queue.shift(); next !== undefined && !this.stopped; next = this.queue.shift()) {
                const { command, context, dataset_id } = next;
                const result =
                    contextOf(this.session()) === context
                        ? await this.run(command, dataset_id)
                        : {
                              request_id: command.request_id,
                              ok: false,
                              error: 'The recording page moved to another session or dataset.',
                          };
                this.running.delete(command.request_id);
                if (!this.stopped) {
                    this.results.set(command.request_id, result);
                }
            }
        } finally {
            this.working = false;
        }
    }
}

const exchangeThroughStudio =
    (project_id: string, robot_id: string) =>
    async (body: SchemaSimulationRelayRequest): Promise<SchemaSimulationRelayReply> => {
        const { data, error } = await fetchClient.POST(
            '/api/projects/{project_id}/robots/{robot_id}/simulation/recording-relay',
            { params: { path: { project_id, robot_id } }, body }
        );
        if (data === undefined) {
            const message = (error as { message?: unknown } | undefined)?.message;
            throw new Error(typeof message === 'string' ? message : 'The simulation did not answer.');
        }
        return data;
    };

export type Relay = {
    /** The simulation's automatic recording, while its simulation runs. */
    recording: SchemaSimulationRecording | null;
    /** Whether a run is under way, so this page relays it. */
    active: boolean;
    /** Why relaying fails, or another page relays the run; ``null`` while it goes well. */
    problem: string | null;
};

/**
 * Relay the automatic recording of the robot's simulation while it runs.
 *
 * The simulation decides when an episode starts and whether it is kept; this page tells it about
 * the session and runs what it asks for, through the session's own commands.
 */
export const useRecordingRelay = ({
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
}): Relay => {
    const { project_id } = useProjectId();
    const { data: simulation } = useRobotSimulation(project_id, robot_id);
    const recording = simulation?.running === true ? (simulation.recording ?? null) : null;
    const active = isRecordingActive(recording);
    const [failure, setFailure] = useState<string | null>(null);
    const [owner, setOwner] = useState(true);

    // Read by the loop, which must not restart on every state message.
    const session = useRef(relaySession(state, follower, leader));
    session.current = relaySession(state, follower, leader);
    const sendRef = useRef(send);
    sendRef.current = send;
    const activeRef = useRef(active);
    activeRef.current = active;
    // One id for as long as the page shows this robot: the simulation hands a run to one page.
    const client = useRef(uuidv4());

    useEffect(() => {
        // A loop per run, kept after the run ends until nothing it asked for is left (a last discard).
        let loop: RecordingRelayLoop | null = null;
        const timer = window.setInterval(() => {
            if (loop === null) {
                if (!activeRef.current) {
                    return;
                }
                loop = new RecordingRelayLoop(
                    client.current,
                    exchangeThroughStudio(project_id, robot_id),
                    (command, dataset_id) => runRelayed(command, dataset_id, sendRef.current),
                    () => session.current,
                    setFailure
                );
            }
            setOwner(loop.owner);
            // After the run: until nothing it asked for is left, or the simulation stays out of reach.
            if (!activeRef.current && (loop.idle || loop.failingFor() > GIVE_UP_MS)) {
                loop.stop();
                loop = null;
                return;
            }
            void loop.tick();
        }, RELAY_INTERVAL_MS);
        return () => {
            loop?.stop();
            window.clearInterval(timer);
        };
    }, [project_id, robot_id]);

    const problem = !active ? null : (failure ?? (owner ? null : 'another recording page relays this run.'));
    return { recording, active, problem };
};

/** How a run goes, in a line: "Recording episode 3: put the cpu away (2 saved, 0 discarded)". */
export const RecordingProgress = ({ recording, active, problem }: Relay) => {
    if (recording == null || recording.phase === 'off') {
        return null;
    }
    const counts = `${recording.saved} saved, ${recording.discarded} discarded`;
    const variant = recording.phase === 'error' || problem ? 'negative' : active ? 'info' : 'neutral';
    return (
        <StatusLight variant={variant}>
            {problem ?? (recording.message || recording.phase)} ({counts})
        </StatusLight>
    );
};

/**
 * Relay a simulation's automatic recording that something else started (a script, with a leader
 * arm teleoperating), and show how it goes.
 */
export const SimulationRecordingRelay = (props: Parameters<typeof useRecordingRelay>[0]) => {
    const relay = useRecordingRelay(props);
    if (relay.recording == null || relay.recording.phase === 'off') {
        return null;
    }
    return (
        <Flex direction='column' gap='size-50'>
            <Text>Recording from the simulation</Text>
            <RecordingProgress {...relay} />
        </Flex>
    );
};
