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
    const setTeleop = async (on: boolean) => {
        const ack = await send({ command: 'set_control', control: on ? TELEOP : null });
        if (!ack.ok) {
            throw new Error(ack.error?.message ?? 'Could not switch teleoperation.');
        }
    };

    const start = useMutation({
        mutationFn: async () => {
            if (!teleop) {
                await setTeleop(true); // the autopilot leads through teleoperation
            }
            await setRecording(project_id, robot_id, {
                ...SETTINGS,
                enabled: true,
                keep,
                max_episodes: episodes,
                ...(perEpisodeInstruction
                    ? { recovery_fraction: stuckStarts, max_attempts: attempts }
                    : { task: task.trim() || defaultTask || null }),
            });
        },
        onError: (error) => toast.negative(error.message),
    });
    const stop = useMutation({
        mutationFn: async () => {
            await setRecording(project_id, robot_id, { ...SETTINGS, enabled: false });
            await setTeleop(false); // also discards the episode being recorded
        },
        onError: (error) => toast.negative(error.message),
    });

    // A run that ended by itself (done, stopped, failed) leaves the arm to the user: teleoperation off.
    const wasActive = useRef(active);
    const sendRef = useRef(send);
    sendRef.current = send;
    useEffect(() => {
        if (wasActive.current && !active && teleop) {
            void sendRef.current({ command: 'set_control', control: null });
        }
        wasActive.current = active;
    }, [active, teleop]);

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
                    Stop
                </Button>
            ) : (
                <Form validationBehavior='native' onSubmit={onStart}>
                    <Flex direction='column' gap='size-150'>
                        {!perEpisodeInstruction && (
                            <TextArea label='Task' width='100%' isRequired value={task} onChange={setTask} />
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
