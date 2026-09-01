import { useState } from 'react';

import { Button, Content, Divider, Flex, Heading, Text, View } from '@geti-ui/ui';

import { RuntimeSessionStatus } from '../features/runtime-sessions/runtime-sessions';
import { queryClient } from '../query-client/query-client';
import { RuntimeSessionFixture, runtimeSessionFixtures, setRuntimeSessionFixture } from './runtime-sessions-fetch-mock';

import classes from './runtime-sessions-preview.module.css';

const applyFixture = (id: RuntimeSessionFixture) => {
    setRuntimeSessionFixture(id);
    void queryClient.invalidateQueries({ queryKey: ['get', '/api/runtime/sessions'] });
    void queryClient.invalidateQueries({ queryKey: ['get', '/api/runtime/sessions/count'] });
};

/**
 * Local playground for the runtime-sessions footer chip and popover.
 * Served only in `rsbuild dev` at /dev/runtime-sessions.
 */
export const RuntimeSessionsPreview = () => {
    const [fixture, setFixture] = useState<RuntimeSessionFixture>('screenshot');

    const onFixture = (id: RuntimeSessionFixture) => {
        setFixture(id);
        applyFixture(id);
    };

    return (
        <div className={classes.page}>
            <header className={classes.header}>
                <Text UNSAFE_className={classes.brand}>Physical AI Studio</Text>
                <Flex gap='size-300'>
                    <Text>Robots</Text>
                    <Text>Datasets</Text>
                    <Text>Models</Text>
                </Flex>
                <Text>runtime</Text>
            </header>

            <View backgroundColor='gray-75' padding='size-300' minHeight={0} overflow='auto'>
                <Content>
                    <Heading level={2}>Runtime sessions UI mock</Heading>
                    <Text>
                        The footer chip on the right is the real widget. Click it, expand a row, and try Stop. Switch
                        fixtures here — no backend required.
                    </Text>
                    <Divider size='S' marginY='size-200' />
                    <Flex gap='size-100' wrap>
                        {runtimeSessionFixtures.map((item) => (
                            <Button
                                key={item.id}
                                variant={fixture === item.id ? 'accent' : 'secondary'}
                                onPress={() => onFixture(item.id)}
                            >
                                {item.label}
                            </Button>
                        ))}
                    </Flex>
                </Content>
            </View>

            <View
                borderTopColor='gray-300'
                borderTopWidth='thin'
                backgroundColor='gray-75'
                paddingX='size-100'
                height='size-400'
            >
                <Flex alignItems='center' height='100%' gap='size-100'>
                    <Text>Logs</Text>
                    <Divider orientation='vertical' size='S' />
                    <RuntimeSessionStatus />
                </Flex>
            </View>
        </div>
    );
};
