import { useEffect, useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Loading,
    StatusLight,
    Text,
    ToastQueue,
    View,
} from '@geti-ui/ui';

import { getApiErrorMessage } from '../../api/errors';
import { SchemaRuntimeSessionInfo } from '../../api/openapi-spec';
import { Table, TableColumn } from '../../components/table/table';
import {
    idleSecondsRemaining,
    sessionActivity,
    sessionLabel,
    uptimeLabel,
    useRuntimeSessionCount,
    useRuntimeSessions,
    useStopRuntimeSession,
} from './use-runtime-sessions';

import classes from './runtime-sessions.module.css';

const COLUMNS: TableColumn[] = [
    { width: '2fr', header: 'Session' },
    { width: '1fr', header: 'Status' },
    { width: '1fr', header: 'Uptime' },
    { width: 'auto', header: '' },
];

const STATUS_VARIANTS: Record<string, 'positive' | 'negative' | 'notice' | 'neutral'> = {
    running: 'positive',
    starting: 'notice',
    stopped: 'neutral',
    error: 'negative',
    unreachable: 'negative',
};

const SessionDetail = ({ session, now }: { session: SchemaRuntimeSessionInfo; now: number }) => {
    const idleSeconds = idleSecondsRemaining(session, now);
    const cameras = session.camera_keys ?? [];
    const facts: string[] = [];

    if (session.activity?.model_loaded) {
        facts.push('Model loaded');
    }
    if (session.activity?.dataset_loaded) {
        facts.push('Dataset loaded');
    }
    if (session.activity?.task) {
        facts.push(`Task: ${session.activity.task}`);
    }
    if (session.activity?.episodes_recorded) {
        facts.push(`${session.activity.episodes_recorded} episodes recorded`);
    }
    if (session.leader_name) {
        facts.push(`Leader: ${session.leader_name}`);
    }
    if (cameras.length > 0) {
        facts.push(`Cameras: ${cameras.join(', ')}`);
    }
    if (session.pid !== null && session.pid !== undefined) {
        facts.push(`pid ${session.pid}`);
    }
    facts.push(session.session_name);

    return (
        <Flex direction='column' gap='size-50' UNSAFE_className={classes.detail}>
            {idleSeconds !== undefined && (
                <Text UNSAFE_className={classes.abandoned}>
                    Nobody is watching this session. It shuts down in {idleSeconds}s.
                </Text>
            )}
            {session.error && (
                <Text UNSAFE_className={classes.errorMessage}>
                    {session.error.message} ({session.error.error_code})
                </Text>
            )}
            {facts.map((fact) => (
                <Text key={fact}>{fact}</Text>
            ))}
        </Flex>
    );
};

const SessionRow = ({
    session,
    now,
    onStop,
}: {
    session: SchemaRuntimeSessionInfo;
    now: number;
    onStop: () => void;
}) => {
    const label = sessionLabel(session);

    return (
        <Table.ExpandableRow label={`Details for ${label}`} detail={<SessionDetail session={session} now={now} />}>
            <Text>{label}</Text>
            <StatusLight variant={STATUS_VARIANTS[session.status] ?? 'neutral'}>{sessionActivity(session)}</StatusLight>
            <Text>{uptimeLabel(session, now) ?? '—'}</Text>
            <Button variant='negative' onPress={onStop} aria-label={`Stop session for ${label}`}>
                Stop
            </Button>
        </Table.ExpandableRow>
    );
};

export const RuntimeSessionsDialog = ({ close }: { close: () => void }) => {
    const { data: sessions, isLoading } = useRuntimeSessions();
    const [stopTarget, setStopTarget] = useState<SchemaRuntimeSessionInfo | undefined>();
    const stopMutation = useStopRuntimeSession();

    // Uptime and the idle countdown are derived from timestamps, so they need a
    // tick of their own -- the poll alone would make them jump in 2s steps.
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        const timer = setInterval(() => setNow(Date.now()), 1_000);
        return () => clearInterval(timer);
    }, []);

    // Owned here rather than inside the confirmation, which unmounts as soon as
    // its primary action fires -- react-query drops the callbacks of an
    // unmounted component, and a failed stop leaves a robot held, so it must not
    // vanish with the dialog.
    const stopSession = (session: SchemaRuntimeSessionInfo) => {
        const label = sessionLabel(session);
        stopMutation.mutate(
            { params: { path: { session_name: session.session_name } } },
            {
                onError: (error) =>
                    ToastQueue.negative(
                        getApiErrorMessage(error) ?? `The session for '${label}' could not be stopped. Try again.`
                    ),
            }
        );
        setStopTarget(undefined);
    };

    return (
        <Dialog onDismiss={close}>
            <Heading>Runtime sessions</Heading>
            <Divider />
            <Content>
                {isLoading ? (
                    <Loading />
                ) : sessions === undefined || sessions.length === 0 ? (
                    <View UNSAFE_className={classes.empty}>
                        <Text>No runtime sessions are running.</Text>
                    </View>
                ) : (
                    <Table columns={COLUMNS}>
                        {sessions.map((session) => (
                            <SessionRow
                                key={session.session_name}
                                session={session}
                                now={now}
                                onStop={() => setStopTarget(session)}
                            />
                        ))}
                    </Table>
                )}

                <DialogContainer onDismiss={() => setStopTarget(undefined)}>
                    {stopTarget && (
                        <AlertDialog
                            title={`Stop session for '${sessionLabel(stopTarget)}'?`}
                            variant='destructive'
                            primaryActionLabel='Stop session'
                            cancelLabel='Cancel'
                            onCancel={() => setStopTarget(undefined)}
                            onPrimaryAction={() => stopSession(stopTarget)}
                            isPrimaryActionDisabled={stopMutation.isPending}
                        >
                            <Text>The session will be terminated.</Text>
                        </AlertDialog>
                    )}
                </DialogContainer>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/**
 * Footer entry point for the runtime sessions running on this host.
 *
 * Renders nothing when none are, matching the job status beside it. Polls the
 * count rather than the list: this is mounted on every page, and the count is a
 * directory read while the list opens a transport session per runtime session.
 */
export const RuntimeSessionStatus = () => {
    const { data } = useRuntimeSessionCount();
    const count = data?.count ?? 0;

    if (count === 0) {
        return null;
    }

    return (
        <DialogTrigger type='fullscreen'>
            <ActionButton isQuiet aria-label='Runtime sessions'>
                <Flex alignItems='center' gap='size-50'>
                    <StatusLight variant='positive' marginEnd='size-0' />
                    <Text>{count === 1 ? '1 session' : `${count} sessions`}</Text>
                </Flex>
            </ActionButton>
            {(close) => <RuntimeSessionsDialog close={close} />}
        </DialogTrigger>
    );
};
