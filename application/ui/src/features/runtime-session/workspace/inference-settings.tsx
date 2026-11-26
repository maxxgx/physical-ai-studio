import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

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

const DEFAULTS: Settings = { task: '', modelId: null, mode: 'sync', threshold: 0.5, hz: 30 };

const STORAGE_KEY_PREFIX = 'physicalai:inferenceSettings:';

const getStorageKey = (projectId: string): string => `${STORAGE_KEY_PREFIX}${projectId}`;

const fits = (value: unknown, { min, max, step }: Range): value is number => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
        return false;
    }
    const steps = (value - min) / step;
    return Math.abs(steps - Math.round(steps)) < 1e-6;
};

const readStored = (projectId: string): Record<string, unknown> => {
    try {
        const stored = JSON.parse(window.localStorage.getItem(getStorageKey(projectId)) ?? '{}');
        return typeof stored === 'object' && stored !== null && !Array.isArray(stored) ? stored : {};
    } catch {
        return {};
    }
};

/** The settings stored for ``projectId``, each one that is missing or does not fit its control at its default. */
export const readInferenceSettings = (projectId: string): Settings => {
    const stored = readStored(projectId);
    return {
        task: typeof stored.task === 'string' ? stored.task : DEFAULTS.task,
        modelId: typeof stored.modelId === 'string' ? stored.modelId : DEFAULTS.modelId,
        mode: stored.mode === 'sync' || stored.mode === 'async' ? stored.mode : DEFAULTS.mode,
        threshold: fits(stored.threshold, THRESHOLD_RANGE) ? stored.threshold : DEFAULTS.threshold,
        hz: fits(stored.hz, HZ_RANGE) ? stored.hz : DEFAULTS.hz,
    };
};

/**
 * Store only the settings that changed, over what is stored now: another tab on the same project
 * keeps the settings it changed, rather than losing them to this tab's older copy.
 */
const writeInferenceSettings = (projectId: string, change: Partial<Settings>): void => {
    try {
        window.localStorage.setItem(getStorageKey(projectId), JSON.stringify({ ...readStored(projectId), ...change }));
    } catch {
        // Storage blocked or full: the settings still hold for this page.
    }
};

/** The default task of the dataset ``model_id`` was trained on: empty if none, undefined until known. */
const useDefaultTask = (model_id: string | undefined): string | undefined => {
    const modelQuery = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: model_id ?? '' } } },
        { enabled: model_id !== undefined }
    );
    const dataset_id = modelQuery.data?.model.dataset_id ?? undefined;
    const datasetQuery = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    if (modelQuery.isError || datasetQuery.isError) {
        return '';
    }
    if (modelQuery.data === undefined) {
        return undefined;
    }
    if (dataset_id === undefined) {
        return '';
    }
    return datasetQuery.data === undefined ? undefined : (datasetQuery.data.default_task ?? '');
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
    const [settings, setSettings] = useState(() => readInferenceSettings(project_id));

    const value = useMemo<InferenceSettings>(() => {
        const update = (change: Partial<Settings>) => {
            setSettings((current) => ({ ...current, ...change }));
            writeInferenceSettings(project_id, change);
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
    }, [project_id, settings]);

    // Another model is loaded and its default task is known: it fills the task if that is empty.
    useEffect(() => {
        if (model_id === undefined || defaultTask === undefined || model_id === settings.modelId) {
            return;
        }
        const change: Partial<Settings> =
            settings.task === '' ? { modelId: model_id, task: defaultTask } : { modelId: model_id };
        setSettings((current) => ({ ...current, ...change }));
        writeInferenceSettings(project_id, change);
    }, [project_id, model_id, defaultTask, settings.modelId, settings.task]);

    return <InferenceSettingsContext.Provider value={value}>{children}</InferenceSettingsContext.Provider>;
};

export const useInferenceSettings = () => {
    const context = useContext(InferenceSettingsContext);
    if (context === null) {
        throw new Error('useInferenceSettings must be used within InferenceSettingsProvider');
    }
    return context;
};
