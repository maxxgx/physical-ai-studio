// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Button,
    DialogTrigger,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Text,
    toast,
    View,
} from '@geti-ui/ui';

import { useProjectId } from '../projects/use-project';
import { startCommandFromMessage, useRobotSimulationLaunch } from './robot-simulation';
import { SimulationDialog } from './simulation-dialog';

import classes from './simulation.module.css';

export const StartCommand = ({ command }: { command: string }) => {
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(command);
            toast.positive('Command copied.');
        } catch {
            toast.negative('Could not copy the command.');
        }
    };

    return (
        <div className={classes.command}>
            <code>{command}</code>
            <ActionButton isQuiet onPress={copy}>
                Copy
            </ActionButton>
        </div>
    );
};

/**
 * Shown where a simulated robot should appear while its simulation is not running: offers to start
 * it, and shows the command to start it by hand when the robot's error message has one.
 */
export const SimulationNotRunning = ({
    robot,
    message,
    onRetry,
    onRunning,
}: {
    robot: { id: string; name: string; type: string };
    message?: string | null;
    onRetry?: () => void;
    onRunning?: () => void;
}) => {
    const { project_id } = useProjectId();
    const launch = useRobotSimulationLaunch(robot.type);
    const command = startCommandFromMessage(message);

    return (
        <View width='100%' height='100%' padding='size-300'>
            <Flex
                width='100%'
                height='100%'
                direction='column'
                justifyContent='center'
                alignItems='center'
                gap='size-200'
            >
                <Heading level={3} margin={0}>
                    Simulation not running
                </Heading>
                <Text>
                    {robot.name} needs a running simulation. Start it to see it here, move its joints and use it in
                    environments.
                </Text>
                <Flex gap='size-100'>
                    {launch !== null ? (
                        <DialogTrigger type='modal'>
                            <Button variant='accent'>Start simulation…</Button>
                            {(close) => (
                                <SimulationDialog
                                    robot={robot}
                                    project_id={project_id}
                                    launch={launch}
                                    close={close}
                                    onRunning={onRunning}
                                />
                            )}
                        </DialogTrigger>
                    ) : null}
                    {onRetry !== undefined ? (
                        <Button variant='secondary' onPress={onRetry}>
                            Retry
                        </Button>
                    ) : null}
                </Flex>
                {command !== undefined ? (
                    <View width='100%' maxWidth='size-6000'>
                        <Disclosure isQuiet>
                            <DisclosureTitle>Or start it yourself</DisclosureTitle>
                            <DisclosurePanel>
                                <StartCommand command={command} />
                            </DisclosurePanel>
                        </Disclosure>
                    </View>
                ) : null}
            </Flex>
        </View>
    );
};
