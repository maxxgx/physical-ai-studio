import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

import { $api } from '../../../api/client';
import { SchemaExecutionConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { useRuntimeSession } from '../runtime-session-provider';

/** What the next Play runs the model with, as the operator last set it. */
interface StoredSettings {
    /** What was typed, or null to follow the loaded model's default task. */
    task: string | null;
    mode: SchemaExecutionConfig['mode'];
    threshold: number;
    // shortcut: set by hand, upgrade to the model's training fps once exports record it.
    hz: number;
}

const DEFAULTS: StoredSettings = { task: null, mode: 'sync', threshold: 0.5, hz: 30 };

const STORAGE_KEY_PREFIX = 'physicalai:inferenceSettings:';

const getStorageKey = (projectId: string): string => `${STORAGE_KEY_PREFIX}${projectId}`;

/** The settings stored for ``projectId``, each one that is missing or invalid at its default. */
export const readInferenceSettings = (projectId: string): StoredSettings => {
    try {
        const stored = JSON.parse(window.localStorage.getItem(getStorageKey(projectId)) ?? '{}');
        return {
            task: typeof stored.task === 'string' ? stored.task : DEFAULTS.task,
            mode: stored.mode === 'sync' || stored.mode === 'async' ? stored.mode : DEFAULTS.mode,
            threshold:
                typeof stored.threshold === 'number' && stored.threshold >= 0 && stored.threshold <= 1
                    ? stored.threshold
                    : DEFAULTS.threshold,
            hz: typeof stored.hz === 'number' && stored.hz > 0 ? stored.hz : DEFAULTS.hz,
        };
    } catch {
        return DEFAULTS;
    }
};

const writeInferenceSettings = (projectId: string, settings: StoredSettings): void => {
    try {
        window.localStorage.setItem(getStorageKey(projectId), JSON.stringify(settings));
    } catch {
        // Storage blocked or full: the settings still hold for this page.
    }
};

/** The default task of the dataset the loaded model was trained on, or none. */
const useDefaultTask = (model_id: string | undefined) => {
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
    return dataset?.default_task ?? '';
};

interface InferenceSettings {
    /** What was typed, or the loaded model's default task until something is. */
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
 * Loading another model never replaces a typed task; only an empty one takes the model's default.
 */
export const InferenceSettingsProvider = ({ children }: { children: ReactNode }) => {
    const { project_id } = useProjectId();
    const { state } = useRuntimeSession();
    const defaultTask = useDefaultTask(state?.model?.model_id);
    const [settings, setSettings] = useState(() => readInferenceSettings(project_id));

    useEffect(() => writeInferenceSettings(project_id, settings), [project_id, settings]);

    // A cleared task follows the default again, so the next model's default fills it.
    useEffect(() => {
        setSettings((current) => (current.task === '' ? { ...current, task: null } : current));
    }, [defaultTask]);

    const value = useMemo<InferenceSettings>(() => {
        const update = (change: Partial<StoredSettings>) => setSettings((current) => ({ ...current, ...change }));
        return {
            task: settings.task ?? defaultTask,
            setTask: (task) => update({ task }),
            mode: settings.mode,
            setMode: (mode) => update({ mode }),
            threshold: settings.threshold,
            setThreshold: (threshold) => update({ threshold }),
            hz: settings.hz,
            setHz: (hz) => update({ hz }),
        };
    }, [settings, defaultTask]);

    return <InferenceSettingsContext.Provider value={value}>{children}</InferenceSettingsContext.Provider>;
};

export const useInferenceSettings = () => {
    const context = useContext(InferenceSettingsContext);
    if (context === null) {
        throw new Error('useInferenceSettings must be used within InferenceSettingsProvider');
    }
    return context;
};
