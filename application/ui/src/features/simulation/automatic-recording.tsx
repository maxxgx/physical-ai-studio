// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useEffect, useRef, useState } from 'react';

import {
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Form,
    NumberField,
    Radio,
    RadioGroup,
    TextArea,
    toast,
} from '@geti-ui/ui';
import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import {
    SchemaSessionState,
    SchemaSimulationRecordingRequest,
    SchemaTeleopControlConfig,
} from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { RecordingProgress, Send, useRecordingRelay } from './recording-relay';

type Keep = 'perfect' | 'all';
type SetControl = (command: { command: 'set_control'; control: SchemaTeleopControlConfig | null }) => ReturnType<Send>;

const STUCK_STARTS_HELP = 'Share of episodes that start with the hand closed and empty on the way, to learn to recover';

/** As the Teleoperate switch turns it on. */
const TELEOP = { type: 'teleop', hz: 100, grace_s: 2 } as const satisfies SchemaTeleopControlConfig;
/** The plugin's defaults, sent with a stop too. */
const SETTINGS = { keep: 'perfect', max_episodes: 0, recovery_fraction: 0, max_attempts: 2 } as const;

const setRecording = async (project_id: string, robot_id: string, body: SchemaSimulationRecordingRequest) => {
    const { error } = await fetchClient.POST('/api/projects/{project_id}/robots/{robot_id}/simulation/recording', {
        params: { path: { project_id, robot_id } },
        body,
    });
    if (error !== undefined) {
        const message = (error as { message?: unknown }).message;
        throw new Error(typeof message === 'string' ? message : 'The simulation did not answer.');
    }
};

/**
 * Record a simulated arm the autopilot leads, automatically: the simulation starts each episode,
 * ends it on its outcome and keeps the good ones; this page runs it through the session.
 *
 * Starting turns teleoperation on (the autopilot leads through it), stopping turns it off, which
 * discards an episode being recorded. Teleoperation also goes off once a run ends by itself.
 */
export const AutomaticRecording = ({
    robot_id,
    state,
    follower,
    leader,
    defaultTask,
    canRecord,
    send,
}: {
    robot_id: string;
    state: SchemaSessionState | null;
    follower: string | null;
    leader: string | null;
    defaultTask: string;
    /** A dataset loaded and the environment working. */
    canRecord: boolean;
    send: Send & SetControl;
}) => {
    const { project_id } = useProjectId();
    const relay = useRecordingRelay({ robot_id, state, follower, leader, send });
    const { recording, active } = relay;
    const [episodes, setEpisodes] = useState(0);
    const [keep, setKeep] = useState<Keep>('perfect');
    const [task, setTask] = useState(defaultTask);
    const [stuckStarts, setStuckStarts] = useState(0);
    const [attempts, setAttempts] = useState(2);
    const perEpisodeInstruction = recording?.task_source === 'instruction';

    useEffect(() => setTask(defaultTask), [defaultTask]);

    const teleop = state?.control?.type === 'teleop';
    // What a start that fails late may undo: only while this panel shows the session it started on.
    const mounted = useRef(true);
    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
        };
    }, []);
    const stateRef = useRef(state);
    stateRef.current = state;
    const setTeleop = async (on: boolean) => {
        const ack = await send({ command: 'set_control', control: on ? TELEOP : null });
        if (!ack.ok) {
            throw new Error(ack.error?.message ?? 'Could not switch teleoperation.');
        }
    };

    const start = useMutation({
        mutationFn: async () => {
            const turnedOn = !teleop;
            const session_id = state?.session_id;
            if (turnedOn) {
                await setTeleop(true); // the autopilot leads through teleoperation
            }
            try {
                await setRecording(project_id, robot_id, {
                    ...SETTINGS,
                    enabled: true,
                    keep,
                    max_episodes: episodes,
                    ...(perEpisodeInstruction
                        ? { recovery_fraction: stuckStarts, max_attempts: attempts }
                        : { task: task.trim() || defaultTask || null }),
                });
            } catch (error) {
                // Nothing records: leave the arm as it was. A start that reached the simulation after all
                // is switched off too.
                await setRecording(project_id, robot_id, { ...SETTINGS, enabled: false }).catch(() => undefined);
                // Not if the user moved on meanwhile (another leader, inference): that control is theirs.
                const current = stateRef.current;
                if (
                    turnedOn &&
                    mounted.current &&
                    current?.session_id === session_id &&
                    current?.control?.type === 'teleop' &&
                    !current?.dataset?.is_recording
                ) {
                    await setTeleop(false).catch(() => undefined);
                }
                throw error;
            }
        },
        onError: (error) => toast.negative(error.message),
    });
    const stop = useMutation({
        mutationFn: async () => {
            // Both, whichever fails: switching teleoperation off holds the arm and discards the episode.
            const results = await Promise.allSettled([
                setRecording(project_id, robot_id, { ...SETTINGS, enabled: false }),
                setTeleop(false),
            ]);
            const failed = results.find((result) => result.status === 'rejected');
            if (failed !== undefined) {
                throw failed.reason instanceof Error ? failed.reason : new Error('Could not stop recording.');
            }
        },
        onError: (error) => toast.negative(error.message),
    });

    // A run the simulation reports over (done, stopped, failed) leaves the arm to the user: teleoperation
    // off. Only on its word: a run whose status is missing for a while, or whose simulation does not
    // answer for a while, is not over, and is still remembered when its end is reported.
    const runSeen = useRef(false);
    const sendRef = useRef(send);
    sendRef.current = send;
    const ended = recording != null && !active;
    useEffect(() => {
        if (active) {
            runSeen.current = true;
        } else if (ended && runSeen.current) {
            runSeen.current = false;
            if (teleop) {
                void sendRef.current({ command: 'set_control', control: null });
            }
        }
    }, [active, ended, teleop]);

    if (recording != null && !recording.available) {
        return null; // the scene records by hand only; the session's own controls stay
    }

    const onStart = (event: FormEvent) => {
        event.preventDefault();
        start.mutate();
    };

    return (
        <Flex direction='column' gap='size-150'>
            <RecordingProgress {...relay} />
            {active ? (
                <Button variant='negative' isPending={stop.isPending} onPress={() => stop.mutate()}>
                    Stop recording
                </Button>
            ) : (
                <Form validationBehavior='native' onSubmit={onStart}>
                    <Flex direction='column' gap='size-150'>
                        {!perEpisodeInstruction && (
                            <TextArea
                                label='Task'
                                width='100%'
                                isRequired
                                maxLength={200}
                                value={task}
                                onChange={setTask}
                            />
                        )}
                        <NumberField
                            label='Episodes'
                            description='0 records until you stop it'
                            minValue={0}
                            maxValue={10_000}
                            step={1}
                            value={episodes}
                            onChange={(value) => setEpisodes(Number.isNaN(value) ? 0 : value)}
                            width='100%'
                        />
                        <RadioGroup
                            label='Keep'
                            orientation='horizontal'
                            value={keep}
                            onChange={(value) => setKeep(value === 'all' ? 'all' : 'perfect')}
                        >
                            <Radio value='perfect'>Perfect only</Radio>
                            <Radio value='all'>All</Radio>
                        </RadioGroup>
                        {perEpisodeInstruction && (
                            <Disclosure isQuiet>
                                <DisclosureTitle>More</DisclosureTitle>
                                <DisclosurePanel>
                                    <Flex direction='column' gap='size-150'>
                                        <NumberField
                                            label='Stuck starts'
                                            description={STUCK_STARTS_HELP}
                                            minValue={0}
                                            maxValue={0.5}
                                            step={0.05}
                                            formatOptions={{ style: 'percent' }}
                                            value={stuckStarts}
                                            onChange={(value) => setStuckStarts(Number.isNaN(value) ? 0 : value)}
                                            width='100%'
                                        />
                                        <NumberField
                                            label='Attempts per episode'
                                            minValue={1}
                                            maxValue={5}
                                            step={1}
                                            value={attempts}
                                            onChange={(value) => setAttempts(Number.isNaN(value) ? 2 : value)}
                                            width='100%'
                                        />
                                    </Flex>
                                </DisclosurePanel>
                            </Disclosure>
                        )}
                        <Button
                            type='submit'
                            variant='accent'
                            isPending={start.isPending}
                            isDisabled={!canRecord || recording == null}
                        >
                            Record automatically
                        </Button>
                    </Flex>
                </Form>
            )}
        </Flex>
    );
};
