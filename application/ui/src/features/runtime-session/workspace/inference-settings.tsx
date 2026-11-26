import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExecutionConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { useRuntimeSession } from '../runtime-session-provider';

interface Range {
    min: number;
    max: number;
    step: number;
}

/** The action rate slider's range, in Hz. */
export const HZ_RANGE: Range = { min: 5, max: 100, step: 5 };
/** The chunk request threshold slider's range, as the fraction of the chunk left. */
export const THRESHOLD_RANGE: Range = { min: 0, max: 1, step: 0.05 };

/** What the next Play runs the model with, as the operator last set it. */
interface Settings {
    task: string;
    /** Whether the operator set the task: a task a fixed-task model picks by itself does not replace it. */
    taskEdited: boolean;
    /** The model last loaded. */
    modelId: string | null;
    /** Whether the model last loaded still owes its default task: it fills the task once known, if empty. */
    fillPending: boolean;
    mode: SchemaExecutionConfig['mode'];
    threshold: number;
    // shortcut: set by hand, upgrade to the model's training fps once exports record it.
    hz: number;
}

type Field = keyof Settings;

const DEFAULTS: Settings = {
    task: '',
    taskEdited: false,
    modelId: null,
    fillPending: false,
    mode: 'sync',
    threshold: 0.5,
    hz: 30,
};
const FIELDS = Object.keys(DEFAULTS) as Field[];

const fits = (value: unknown, { min, max, step }: Range): value is number => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
        return false;
    }
    const steps = (value - min) / step;
    return Math.abs(steps - Math.round(steps)) < 1e-6;
};

/** Whether a stored value is one the field can take; anything else reads as the default. */
const VALID: { [F in Field]: (value: unknown) => value is Settings[F] } = {
    task: (value): value is string => typeof value === 'string',
    taskEdited: (value): value is boolean => typeof value === 'boolean',
    modelId: (value): value is string => typeof value === 'string',
    fillPending: (value): value is boolean => typeof value === 'boolean',
    mode: (value): value is Settings['mode'] => value === 'sync' || value === 'async',
    threshold: (value): value is number => fits(value, THRESHOLD_RANGE),
    hz: (value): value is number => fits(value, HZ_RANGE),
};

/**
 * The settings of one project in this tab's session storage, which a reload keeps. They are read
 * once, when the workspace opens the project; after that the page holds them and only writes, so a
 * write that fails never brings back an older value. A value that cannot be replaced is removed,
 * so a later workspace does not start from it either.
 */
const projectStorage = (projectId: string) => {
    const key = (field: Field) => `physicalai:inferenceSettings:${projectId}:${field}`;

    /** The stored settings, each missing or invalid one at its default, and whether storage failed. */
    const readAll = (): { settings: Settings; failed: boolean } => {
        const settings = { ...DEFAULTS };
        for (const field of FIELDS) {
            let raw: string | null;
            try {
                raw = window.sessionStorage.getItem(key(field));
            } catch {
                return { settings: DEFAULTS, failed: true };
            }
            try {
                const value: unknown = raw === null ? undefined : JSON.parse(raw);
                if (VALID[field](value)) {
                    Object.assign(settings, { [field]: value });
                }
            } catch {
                // Not JSON: the default stays.
            }
        }
        return { settings, failed: false };
    };

    /** Store ``change``; false if a value could not be stored. */
    const write = (change: Partial<Settings>): boolean => {
        let stored = true;
        for (const [field, value] of Object.entries(change)) {
            try {
                window.sessionStorage.setItem(key(field as Field), JSON.stringify(value));
            } catch {
                stored = false;
                try {
                    window.sessionStorage.removeItem(key(field as Field));
                } catch {
                    // Blocked altogether: nothing stored to go stale.
                }
            }
        }
        return stored;
    };

    return { readAll, write };
};

/**
 * The default task of the dataset ``model_id`` was trained on: empty if none, undefined while
 * it is not known, which includes a failed fetch: a reload or the next load tries again.
 */
const useDefaultTask = (model_id: string | undefined): string | undefined => {
    const { data: model } = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: model_id ?? '' } } },
        { enabled: model_id !== undefined }
    );
    const dataset_id = model?.model.dataset_id ?? undefined;
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    if (model === undefined) {
        return undefined;
    }
    if (dataset_id === undefined) {
        return '';
    }
    return dataset === undefined ? undefined : (dataset.default_task ?? '');
};

interface InferenceSettings {
    task: string;
    /** The operator set the task. */
    setTask: (task: string) => void;
    /** A task picked for the operator, such as a fixed-task model's first one: kept unless they set one. */
    offerTask: (task: string) => void;
    mode: SchemaExecutionConfig['mode'];
    setMode: (mode: SchemaExecutionConfig['mode']) => void;
    threshold: number;
    setThreshold: (threshold: number) => void;
    hz: number;
    setHz: (hz: number) => void;
}

const InferenceSettingsContext = createContext<InferenceSettings | null>(null);

const STORAGE_FAILED_MESSAGE =
    'This browser does not let Studio store the inference settings: they will not be kept after a reload.';

/**
 * The task and execution settings for the whole workspace, kept per project in this tab.
 *
 * They live above the mode panels, which unmount on every mode switch, and are stored so that a
 * reload starts from what the operator last set, not from Studio's defaults. A model's default
 * fills the task only when it is empty, once per model loaded, so neither a model swap nor a
 * reload replaces a task.
 */
export const InferenceSettingsProvider = ({ children }: { children: ReactNode }) => {
    const { project_id } = useProjectId();
    const { state } = useRuntimeSession();
    const model_id = state?.model?.model_id;
    const defaultTask = useDefaultTask(model_id);

    const storage = useMemo(() => projectStorage(project_id), [project_id]);
    const [opened, setOpened] = useState(() => ({ project_id, ...storage.readAll() }));
    let current = opened;
    if (opened.project_id !== project_id) {
        // Another project opened in place: start from its settings, not the last project's.
        const read = storage.readAll();
        current = { project_id, settings: read.settings, failed: opened.failed || read.failed };
        setOpened(current);
    }
    const { settings, failed } = current;

    const update = useCallback(
        (change: Partial<Settings>) => {
            const stored = storage.write(change);
            setOpened((open) => ({
                ...open,
                settings: { ...open.settings, ...change },
                failed: open.failed || !stored,
            }));
        },
        [storage]
    );

    // Once per workspace, and only for one that is shown.
    const warned = useRef(false);
    useEffect(() => {
        if (failed && !warned.current) {
            warned.current = true;
            toast.info(STORAGE_FAILED_MESSAGE);
        }
    }, [failed]);

    const value = useMemo<InferenceSettings>(
        () => ({
            task: settings.task,
            setTask: (task) => update({ task, taskEdited: true }),
            offerTask: (task) => {
                if (!settings.taskEdited) {
                    update({ task });
                }
            },
            mode: settings.mode,
            setMode: (mode) => update({ mode }),
            threshold: settings.threshold,
            setThreshold: (threshold) => update({ threshold }),
            hz: settings.hz,
            setHz: (hz) => update({ hz }),
        }),
        [update, settings]
    );

    // Each model loaded owes its default task, which fills the task once known if the task is empty.
    // A failed fetch leaves it owed, for a reload or the next load of that model.
    useEffect(() => {
        if (model_id === undefined) {
            return;
        }
        if (model_id !== settings.modelId) {
            update({ modelId: model_id, fillPending: true });
        } else if (settings.fillPending && defaultTask !== undefined) {
            update(
                settings.task === ''
                    ? { task: defaultTask, taskEdited: false, fillPending: false }
                    : { fillPending: false }
            );
        }
    }, [update, model_id, defaultTask, settings.modelId, settings.fillPending, settings.task]);

    return <InferenceSettingsContext.Provider value={value}>{children}</InferenceSettingsContext.Provider>;
};

export const useInferenceSettings = () => {
    const context = useContext(InferenceSettingsContext);
    if (context === null) {
        throw new Error('useInferenceSettings must be used within InferenceSettingsProvider');
    }
    return context;
};
