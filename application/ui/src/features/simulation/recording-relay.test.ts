// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from '@rstest/core';

import {
    SchemaSimulationRelayCommand,
    SchemaSimulationRelayReply,
    SchemaSimulationRelayRequest,
    SchemaSimulationRelayResult,
    SchemaSimulationRelaySession,
} from '../../api/openapi-spec';
import { RecordingRelayLoop, RETRY_MS } from './recording-relay';

const SESSION: SchemaSimulationRelaySession = {
    session_id: 's1',
    dataset_id: 'd1',
    dataset: true,
    recording: false,
    teleop: true,
    follower: 'Sim arm',
    leader: 'Sim arm autopilot',
};
const SAVE: SchemaSimulationRelayCommand = { command: 'save_episode', request_id: 'save-1' };
const ok = (request_id: string): SchemaSimulationRelayResult => ({ request_id, ok: true, error: null });

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A loop against scripted replies; `run` resolves when the test says so. */
const setUp = (replies: SchemaSimulationRelayReply[]) => {
    const requests: SchemaSimulationRelayRequest[] = [];
    const runs: SchemaSimulationRelayCommand[] = [];
    const datasets: (string | null)[] = [];
    const finish: (() => void)[] = [];
    const failures: (string | null)[] = [];
    let session = SESSION;
    let now = 0;
    let fail = false;
    const loop = new RecordingRelayLoop(
        'page-1',
        async (request) => {
            requests.push(request);
            if (fail) {
                throw new Error('The simulation does not answer');
            }
            return replies[Math.min(requests.length - 1, replies.length - 1)];
        },
        (command, dataset_id) =>
            new Promise((resolve) => {
                runs.push(command);
                datasets.push(dataset_id);
                finish.push(() => resolve(ok(command.request_id)));
            }),
        () => session,
        (message) => failures.push(message),
        () => now
    );
    return {
        loop,
        requests,
        runs,
        datasets,
        finish,
        failures,
        tick: async () => {
            await loop.tick();
            await settle();
        },
        moveTo: (next: SchemaSimulationRelaySession) => (session = next),
        advance: (ms: number) => (now += ms),
        failExchanges: (value: boolean) => (fail = value),
    };
};

describe('RecordingRelayLoop', () => {
    it('runs a command once, and sends its result until the simulation stops listing it', async () => {
        const listed = { active: true, owner: true, commands: [SAVE] };
        const relay = setUp([listed, listed, listed, { active: true, owner: true, commands: [] }]);

        await relay.tick();
        relay.finish[0]();
        await settle();
        await relay.tick(); // the reply to this one was lost: the command comes again
        await relay.tick();
        await relay.tick();

        expect(relay.runs).toEqual([SAVE]);
        expect(relay.datasets).toEqual(['d1']); // the session refuses it if another dataset was loaded meanwhile
        expect(relay.requests.map(({ results }) => results)).toEqual([
            [],
            [ok('save-1')],
            [ok('save-1')],
            [ok('save-1')],
        ]);
        await relay.tick();
        expect(relay.requests.at(-1)?.results).toEqual([]);
    });

    it('keeps relaying while a slow save runs', async () => {
        const relay = setUp([{ active: true, owner: true, commands: [SAVE] }]);

        for (let i = 0; i < 50; i++) {
            await relay.tick();
        }

        expect(relay.requests).toHaveLength(50);
        expect(relay.runs).toEqual([SAVE]);
    });

    it('keeps the results of a failed exchange, says why and waits before trying again', async () => {
        const relay = setUp([{ active: true, owner: true, commands: [SAVE] }]);
        await relay.tick();
        relay.finish[0]();
        await settle();

        relay.failExchanges(true);
        await relay.tick();
        await relay.tick(); // too soon
        expect(relay.requests).toHaveLength(2);
        expect(relay.failures.at(-1)).toBe('The simulation does not answer');

        relay.failExchanges(false);
        relay.advance(RETRY_MS);
        await relay.tick();
        expect(relay.requests.at(-1)?.results).toEqual([ok('save-1')]);
        expect(relay.failures.at(-1)).toBeNull();
    });

    it('does not run a command on another session or dataset than it came on', async () => {
        const start: SchemaSimulationRelayCommand = { command: 'start_episode', request_id: 'start-1', task: 'Sort' };
        const relay = setUp([{ active: true, owner: true, commands: [SAVE, start] }]);
        await relay.tick();
        relay.moveTo({ ...SESSION, dataset_id: 'd2' }); // the user loaded another dataset meanwhile
        relay.finish[0]();
        await settle();
        await relay.tick();

        expect(relay.runs).toEqual([SAVE]);
        expect(relay.requests.at(-1)?.results).toEqual([
            ok('save-1'),
            { request_id: 'start-1', ok: false, error: 'The recording page moved to another session or dataset.' },
        ]);
    });

    it('does not run a command the simulation withdrew before it started', async () => {
        const discard: SchemaSimulationRelayCommand = { command: 'discard_episode', request_id: 'last' };
        const start: SchemaSimulationRelayCommand = { command: 'start_episode', request_id: 'next', task: 'Sort' };
        const relay = setUp([
            { active: true, owner: true, commands: [discard, start] },
            { active: false, owner: true, commands: [discard] }, // the new run was switched off
            { active: false, owner: true, commands: [] },
        ]);
        await relay.tick(); // the discard runs, the start waits behind it
        await relay.tick();
        relay.finish[0]();
        await settle();
        await relay.tick();

        expect(relay.runs).toEqual([discard]);
        expect(relay.loop.idle).toBe(true);
    });

    it('counts how long the simulation has been out of reach', async () => {
        const relay = setUp([{ active: true, owner: true, commands: [] }]);
        await relay.tick();
        expect(relay.loop.failingFor()).toBe(0);

        relay.failExchanges(true);
        relay.advance(RETRY_MS);
        await relay.tick();
        relay.advance(RETRY_MS * 3);
        await relay.tick();
        expect(relay.loop.failingFor()).toBe(RETRY_MS * 3);

        relay.failExchanges(false);
        relay.advance(RETRY_MS);
        await relay.tick();
        expect(relay.loop.failingFor()).toBe(0);
    });

    it('is idle once the run ended and nothing is left, and stops running commands once stopped', async () => {
        const relay = setUp([
            { active: false, owner: true, commands: [{ command: 'discard_episode', request_id: 'last' }] },
            { active: false, owner: true, commands: [] },
        ]);
        await relay.tick();
        expect(relay.loop.idle).toBe(false);
        relay.finish[0]();
        await settle();
        await relay.tick();
        expect(relay.loop.idle).toBe(true);

        relay.loop.stop();
        await relay.tick();
        expect(relay.requests).toHaveLength(2);
    });
});
