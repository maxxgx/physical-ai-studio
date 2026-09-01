import { lazy, Suspense, useState, type CSSProperties } from 'react';

import { Button, Content, Divider, Flex, Heading, Text, View } from '@geti-ui/ui';

import type { SchemaRuntimeSessionInfo } from '../src/api/openapi-spec';
import { queryClient } from '../src/query-client/query-client';

// Loaded after this module patches fetch. A static import would pull in $api
// first, and openapi-fetch would capture the real fetch before the stub is in.
const RuntimeSessionStatus = lazy(() =>
    import('../src/features/runtime-sessions/runtime-sessions').then((mod) => ({
        default: mod.RuntimeSessionStatus,
    }))
);

type RuntimeSessionFixture = 'screenshot' | 'two' | 'recording' | 'abandoned' | 'unreachable' | 'empty';

const ROBOT_A = '0ff6cb53-5a72-4be0-a672-a95520ffb3b3';
const ROBOT_B = '9c1f5b8e-2f4a-4c6e-8a11-7d2e3f4a5b60';

type Fixtures = Record<RuntimeSessionFixture, () => SchemaRuntimeSessionInfo[]>;

// Everything below is referenced only from inside the `import.meta.env.DEV`
// block, so a production build folds the flag to false and drops the fixture
// bodies entirely. Keep it that way: at module scope they survive tree-shaking
// and ship fake robots to real users.
let sessions: SchemaRuntimeSessionInfo[] = [];
let fixtures: Fixtures | null = null;

const activity = (
    overrides: Partial<NonNullable<SchemaRuntimeSessionInfo['activity']>> = {}
): NonNullable<SchemaRuntimeSessionInfo['activity']> => ({
    connected: true,
    follower_source: 'hold',
    model_loaded: false,
    task: null,
    dataset_loaded: false,
    is_recording: false,
    episodes_recorded: 0,
    ...overrides,
});

const session = (overrides: Partial<SchemaRuntimeSessionInfo> = {}): SchemaRuntimeSessionInfo => ({
    session_name: `rt-${ROBOT_A}`,
    follower_id: ROBOT_A,
    status: 'running',
    pid: 3478727,
    follower_name: 't1000',
    leader_name: 'captain',
    started_at: new Date(Date.now() - 5 * 60_000).toISOString(),
    idle_timeout_s: 45,
    attached: true,
    idle_deadline: null,
    camera_keys: ['gripper', 'overview'],
    activity: activity(),
    error: null,
    ...overrides,
});

const buildFixtures = (): Fixtures => ({
    screenshot: () => [session()],
    two: () => [
        session(),
        session({
            session_name: `rt-${ROBOT_B}`,
            follower_id: ROBOT_B,
            follower_name: 'left arm',
            leader_name: 'left leader',
            pid: 41273,
            camera_keys: ['wrist'],
            activity: activity({ follower_source: 'teleop', dataset_loaded: true, task: 'pick up the cube' }),
            started_at: new Date(Date.now() - 90_000).toISOString(),
        }),
    ],
    recording: () => [
        session({
            activity: activity({
                follower_source: 'teleop',
                is_recording: true,
                dataset_loaded: true,
                episodes_recorded: 3,
                task: 'pick up the cube',
            }),
        }),
    ],
    abandoned: () => [
        session({
            attached: false,
            idle_deadline: new Date(Date.now() + 30_000).toISOString(),
        }),
    ],
    unreachable: () => [
        session({
            status: 'unreachable',
            follower_name: null,
            leader_name: null,
            activity: null,
            camera_keys: [],
            error: { message: 'Transport closed', error_code: 'unreachable' },
        }),
    ],
    empty: () => [],
});

const runtimeSessionFixtures: { id: RuntimeSessionFixture; label: string }[] = import.meta.env.DEV
    ? [
          { id: 'screenshot', label: 'One session (t1000)' },
          { id: 'two', label: 'Two sessions' },
          { id: 'recording', label: 'Recording' },
          { id: 'abandoned', label: 'Abandoned' },
          { id: 'unreachable', label: 'Unreachable' },
          { id: 'empty', label: 'None running' },
      ]
    : [];

const setRuntimeSessionFixture = (id: RuntimeSessionFixture) => {
    sessions = fixtures?.[id]() ?? [];
};

const requestUrl = (input: RequestInfo | URL): URL => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    return new URL(raw, window.location.origin);
};

const requestMethod = (input: RequestInfo | URL, init?: RequestInit): string => {
    if (init?.method) {
        return init.method.toUpperCase();
    }
    if (input instanceof Request) {
        return input.method.toUpperCase();
    }
    return 'GET';
};

const isPreviewPath = () => window.location.pathname.startsWith('/mockups/runtime-sessions');

/**
 * Patch fetch before the OpenAPI client captures it. Only answers runtime-session
 * routes while the local preview is open; everything else passes through.
 */
if (import.meta.env.DEV) {
    fixtures = buildFixtures();
    sessions = fixtures.screenshot();

    const originalFetch = globalThis.fetch.bind(globalThis);

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        if (!isPreviewPath()) {
            return originalFetch(input, init);
        }

        const url = requestUrl(input);
        const method = requestMethod(input, init);
        const path = url.pathname;

        if (path.endsWith('/api/runtime/sessions/count') && method === 'GET') {
            return Response.json({ count: sessions.length });
        }

        const stopMatch = path.match(/\/api\/runtime\/sessions\/([^/]+)\/stop$/);
        if (stopMatch && method === 'POST') {
            const sessionName = decodeURIComponent(stopMatch[1]);
            sessions = sessions.filter((item) => item.session_name !== sessionName);
            return new Response(null, { status: 204 });
        }

        if (path.endsWith('/api/runtime/sessions') && method === 'GET') {
            return Response.json(sessions);
        }

        return originalFetch(input, init);
    };
}

const applyFixture = (id: RuntimeSessionFixture) => {
    setRuntimeSessionFixture(id);
    void queryClient.invalidateQueries({ queryKey: ['get', '/api/runtime/sessions'] });
    void queryClient.invalidateQueries({ queryKey: ['get', '/api/runtime/sessions/count'] });
};

const pageStyle: CSSProperties = {
    display: 'grid',
    gridTemplateRows: 'auto 1fr var(--spectrum-global-dimension-size-400)',
    height: '100%',
    minHeight: '100%',
};

const headerStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--spectrum-global-dimension-size-200)',
    padding: '0 var(--spectrum-global-dimension-size-200)',
    height: 'var(--spectrum-global-dimension-size-600)',
    background: 'var(--spectrum-global-color-gray-100)',
    borderBottom: '1px solid var(--spectrum-global-color-gray-300)',
};

/**
 * Local playground for the runtime-sessions footer chip and popover.
 * Served only in `rsbuild dev` at /mockups/runtime-sessions.
 */
export const RuntimeSessionsPreview = () => {
    const [fixture, setFixture] = useState<RuntimeSessionFixture>('screenshot');

    const onFixture = (id: RuntimeSessionFixture) => {
        setFixture(id);
        applyFixture(id);
    };

    return (
        <div style={pageStyle}>
            <header style={headerStyle}>
                <Text UNSAFE_style={{ fontWeight: 600 }}>Physical AI Studio</Text>
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
                    <Suspense>
                        <RuntimeSessionStatus />
                    </Suspense>
                </Flex>
            </View>
        </div>
    );
};
