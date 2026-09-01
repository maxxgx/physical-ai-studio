import { SchemaRuntimeSessionInfo } from '../api/openapi-spec';

export type RuntimeSessionFixture = 'screenshot' | 'two' | 'recording' | 'abandoned' | 'unreachable' | 'empty';

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

export const runtimeSessionFixtures: { id: RuntimeSessionFixture; label: string }[] = import.meta.env.DEV
    ? [
          { id: 'screenshot', label: 'One session (t1000)' },
          { id: 'two', label: 'Two sessions' },
          { id: 'recording', label: 'Recording' },
          { id: 'abandoned', label: 'Abandoned' },
          { id: 'unreachable', label: 'Unreachable' },
          { id: 'empty', label: 'None running' },
      ]
    : [];

export const setRuntimeSessionFixture = (id: RuntimeSessionFixture) => {
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

const isPreviewPath = () => window.location.pathname.startsWith('/dev/runtime-sessions');

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
