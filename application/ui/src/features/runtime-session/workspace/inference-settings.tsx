import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

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
    /** The model whose default task was last offered: a model loaded after it fills an empty task. */
    modelId: string | null;
    mode: SchemaExecutionConfig['mode'];
    threshold: number;
    // shortcut: set by hand, upgrade to the model's training fps once exports record it.
    hz: number;
}

type Field = keyof Settings;

const DEFAULTS: Settings = { task: '', modelId: null, mode: 'sync', threshold: 0.5, hz: 30 };

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
    modelId: (value): value is string => typeof value === 'string',
    mode: (value): value is Settings['mode'] => value === 'sync' || value === 'async',
    threshold: (value): value is number => fits(value, THRESHOLD_RANGE),
    hz: (value): value is number => fits(value, HZ_RANGE),
};

/**
 * The settings of one project in this browser, each field under its own key: a tab writes only
 * the fields it changed, so two tabs on one project never write back each other's older values.
 * ``onFailure`` is called when the browser blocks or cannot fit a value.
 */
const projectStorage = (projectId: string, onFailure: () => void) => {
    const key = (field: Field) => `physicalai:inferenceSettings:${projectId}:${field}`;

    /** The stored value of ``field``: its default if missing or invalid, undefined if storage failed. */
    const read = <F extends Field>(field: F): Settings[F] | undefined => {
        let raw: string | null;
        try {
            raw = window.localStorage.getItem(key(field));
        } catch {
            onFailure();
            return undefined;
        }
        try {
            const value: unknown = raw === null ? undefined : JSON.parse(raw);
            return VALID[field](value) ? (value as Settings[F]) : DEFAULTS[field];
        } catch {
            return DEFAULTS[field];
        }
    };

    const write = (change: Partial<Settings>) => {
        for (const [field, value] of Object.entries(change)) {
            try {
                window.localStorage.setItem(key(field as Field), JSON.stringify(value));
            } catch {
                onFailure();
            }
        }
    };

    const readAll = (): Settings => ({
        task: read('task') ?? DEFAULTS.task,
        modelId: read('modelId') ?? DEFAULTS.modelId,
        mode: read('mode') ?? DEFAULTS.mode,
        threshold: read('threshold') ?? DEFAULTS.threshold,
        hz: read('hz') ?? DEFAULTS.hz,
    });

    return { read, write, readAll };
};

let warned = false;

/** Say once per page that the settings will not outlive it: the browser blocks storage for every project. */
const warnOnce = () => {
    if (!warned) {
        warned = true;
        toast.info(
            'This browser does not let Studio store the inference settings: they will not be kept after a reload.'
        );
    }
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
    setTask: (task: string) => void;
    mode: SchemaExecutionConfig['mode'];
    setMode: (mode: SchemaExecutionConfig['mode']) => void;
    threshold: number;
    setThreshold: (threshold: number) => void;
    hz: number;
    setHz: (hz: number) => void;
}

const InferenceSettingsContext = createContext<InferenceSettings | null>(null);

/**
 * The task and execution settings for the whole workspace, kept per project in this browser.
 *
 * They live above the mode panels, which unmount on every mode switch, and are stored so that a
 * page load or a new session starts from what the operator last ran, not from Studio's defaults.
 * The task is what the operator sees: a model's default fills it only when it is empty and that
 * model was not the last one offered, so neither a model swap nor a reload replaces a task.
 */
export const InferenceSettingsProvider = ({ children }: { children: ReactNode }) => {
    const { project_id } = useProjectId();
    const { state } = useRuntimeSession();
    const model_id = state?.model?.model_id;
    const defaultTask = useDefaultTask(model_id);

    const storage = useMemo(() => projectStorage(project_id, warnOnce), [project_id]);
    const [settings, setSettings] = useState(() => storage.readAll());

    const value = useMemo<InferenceSettings>(() => {
        const update = (change: Partial<Settings>) => {
            setSettings((current) => ({ ...current, ...change }));
            storage.write(change);
        };
        return {
            task: settings.task,
            setTask: (task) => update({ task }),
            mode: settings.mode,
            setMode: (mode) => update({ mode }),
            threshold: settings.threshold,
            setThreshold: (threshold) => update({ threshold }),
            hz: settings.hz,
            setHz: (hz) => update({ hz }),
        };
    }, [storage, settings]);

    // Another model is loaded and its default task is known: it fills the task if that is empty.
    // Another tab may have offered this model or set a task meanwhile, so decide from what is stored now.
    useEffect(() => {
        if (model_id === undefined || defaultTask === undefined || model_id === settings.modelId) {
            return;
        }
        const offered = storage.read('modelId') ?? settings.modelId;
        const stored = storage.read('task') ?? settings.task;
        const fill = offered !== model_id && stored === '';
        const task = fill ? defaultTask : stored;
        setSettings((current) => ({ ...current, modelId: model_id, task }));
        if (offered !== model_id) {
            storage.write(fill ? { modelId: model_id, task } : { modelId: model_id });
        }
    }, [storage, model_id, defaultTask, settings.modelId, settings.task]);

    return <InferenceSettingsContext.Provider value={value}>{children}</InferenceSettingsContext.Provider>;
};

export const useInferenceSettings = () => {
    const context = useContext(InferenceSettingsContext);
    if (context === null) {
        throw new Error('useInferenceSettings must be used within InferenceSettingsProvider');
    }
    return context;
};
