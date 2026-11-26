// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export interface LogTime {
    timestamp: number;
    repr: string;
}

export interface LogLevel {
    name: LogLevelName;
    no: number;
    icon: string;
}

export type LogLevelName = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS' | 'CRITICAL';

export const LOG_LEVELS: LogLevelName[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'];

export const LOG_LEVEL_COLORS: Record<LogLevelName, string> = {
    DEBUG: 'var(--spectrum-global-color-gray-600)',
    INFO: '#ffffff',
    WARNING: 'var(--spectrum-global-color-orange-600)',
    ERROR: 'var(--spectrum-global-color-red-600)',
    SUCCESS: '#4ade80',
    CRITICAL: 'var(--spectrum-global-color-magenta-600)',
};

export interface LogProcess {
    id: number;
    name: string;
}

export interface LogThread {
    id: number;
    name: string;
}

export interface LogFile {
    name: string;
    path: string;
}

export interface LogRecord {
    elapsed: { repr: string; seconds: number };
    exception: unknown;
    extra: Record<string, unknown>;
    file: LogFile;
    function: string;
    level: LogLevel;
    line: number;
    message: string;
    module: string;
    name: string;
    process: LogProcess;
    thread: LogThread;
    time: LogTime;
}

export interface LogEntry {
    text: string;
    record: LogRecord;
}

const describe = (value: unknown): string => {
    if (typeof value === 'string') {
        return value;
    }
    try {
        return JSON.stringify(value) ?? '';
    } catch {
        return '';
    }
};

/**
 * The exception of a record: the plain traceback the backend stores in `extra.traceback`, or else the
 * exception's type and value. `null` when the record has none. Built from the structured record, so it
 * does not depend on the text format, and it never throws on a malformed record.
 */
export const getExceptionText = (entry: LogEntry): string | null => {
    const { exception, extra } = entry.record;
    if (exception === null || typeof exception !== 'object') {
        return null;
    }
    const traceback = typeof extra === 'object' && extra !== null ? (extra as Record<string, unknown>).traceback : null;
    if (typeof traceback === 'string' && traceback.trim() !== '') {
        return traceback.trimEnd();
    }
    const { type, value } = exception as { type?: unknown; value?: unknown };
    if (typeof type !== 'string') {
        return null;
    }
    const description = describe(value);
    return description === '' ? type : `${type}: ${description}`;
};

export interface LogFilters {
    levels: Set<LogLevelName>;
    searchQuery: string;
    startTime: number | null;
    endTime: number | null;
}

export const DEFAULT_LOG_FILTERS: LogFilters = {
    levels: new Set(LOG_LEVELS),
    searchQuery: '',
    startTime: null,
    endTime: null,
};

export interface LogSource {
    id: string;
    name: string;
    type: 'application' | 'worker' | 'job';
    /** ISO-8601 creation timestamp of the log file, if available. */
    created_at?: string | null;
}
