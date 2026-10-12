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

import { startCommandFromMessage, StartCommandHint } from './robot-simulation';

import classes from './simulation.module.css';

export const StartCommand = ({ command, host }: StartCommandHint) => {
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(command);
            toast.positive('Command copied.');
        } catch {
            toast.negative('Could not copy the command.');
        }
    };

    return (
        <Flex direction='column' gap='size-50'>
            {host !== undefined ? <Text>Run it on {host}:</Text> : null}
            <div className={classes.command}>
                <code>{command}</code>
                <ActionButton isQuiet onPress={copy}>
                    Copy
                </ActionButton>
            </div>
        </Flex>
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
    onReconnect,
    isStarting = false,
}: {
    robot: { name: string };
    message?: string | null;
    onStart?: () => void;
    /** Connects again, for after the user started the simulation by hand. */
    onReconnect?: () => void;
    /** Studio is starting (or stopping) the simulation: offer its progress instead. */
    isStarting?: boolean;
}) => {
    const hint = startCommandFromMessage(message);

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
                    {isStarting ? 'Simulation starting' : 'Simulation not running'}
                </Heading>
                <Text>
                    {robot.name} needs a running simulation. Start it to see it here, move its joints and use it in
                    environments.
                </Text>
                <Flex gap='size-100'>
                    {onStart !== undefined ? (
                        <Button variant='accent' onPress={onStart}>
                            {isStarting ? 'Show progress…' : 'Start simulation…'}
                        </Button>
                    ) : null}
                </Flex>
                {hint !== undefined ? (
                    <View width='100%' maxWidth='size-6000'>
                        <Disclosure isQuiet>
                            <DisclosureTitle>Or start it yourself</DisclosureTitle>
                            <DisclosurePanel>
                                <Flex direction='column' gap='size-100'>
                                    <StartCommand {...hint} />
                                    {onReconnect !== undefined ? (
                                        <Flex alignItems='center' gap='size-100'>
                                            <Text UNSAFE_className={classes.detail}>Started it?</Text>
                                            <Button variant='secondary' onPress={onReconnect}>
                                                Reconnect
                                            </Button>
                                        </Flex>
                                    ) : null}
                                </Flex>
                            </DisclosurePanel>
                        </Disclosure>
                    </View>
                ) : null}
            </Flex>
        </View>
    );
};
