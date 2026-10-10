// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Text,
    toast,
    View,
} from '@geti-ui/ui';

import { startCommandFromMessage } from './robot-simulation';

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
 * it (`onStart`, from `useSimulationDialog`), and shows the command to start it by hand when the
 * robot's error message has one.
 */
export const SimulationNotRunning = ({
    robot,
    message,
    onStart,
    onRetry,
}: {
    robot: { name: string };
    message?: string | null;
    onStart?: () => void;
    onRetry?: () => void;
}) => {
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
                    {onStart !== undefined ? (
                        <Button variant='accent' onPress={onStart}>
                            Start simulation…
                        </Button>
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
