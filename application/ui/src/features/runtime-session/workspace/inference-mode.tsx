import { useEffect, useState } from 'react';

import { Button, Flex, Item, Picker, Slider, TextArea, toast } from '@geti-ui/ui';

import { SchemaExecutionConfig, SchemaModelControlConfig } from '../../../api/openapi-spec';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';
import { HZ_RANGE, THRESHOLD_RANGE, useInferenceSettings } from './inference-settings';

const EXECUTION_MODES: { key: SchemaExecutionConfig['mode']; label: string }[] = [
    { key: 'sync', label: 'Synchronous' },
    { key: 'async', label: 'Asynchronous' },
];

/** Inference mode's side bar: the task, the action rate and how chunks are requested, then play / stop. */
export const InferenceMode = () => {
    const { state, statuses, send } = useRuntimeSession();
    // Held above this panel, which unmounts on every mode switch.
    const { task, setTask, offerTask, mode, setMode, threshold, setThreshold, hz, setHz } = useInferenceSettings();
    const [isPending, setIsPending] = useState(false);

    // A model trained on a fixed set of tasks only takes one of them.
    const tasks = state?.model?.tasks ?? [];
    const selectedTask = tasks.length > 0 && !tasks.includes(task) ? tasks[0] : task;
    // Keep the one shown as the task, so a free-text model loaded next starts from what ran,
    // unless the operator set a task.
    useEffect(() => {
        if (selectedTask !== task) {
            offerTask(selectedTask);
        }
    }, [selectedTask, task, offerTask]);

    const isRunning = state?.control?.type === 'model';
    const canRun =
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        state.model != null &&
        failureOf(statuses, 'model') === null;

    const setControl = async (control: SchemaModelControlConfig | null) => {
        setIsPending(true);
        try {
            const ack = await send({ command: 'set_control', control });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not run the model.', ERROR_TOAST);
            }
        } finally {
            setIsPending(false);
        }
    };

    const play = () =>
        void setControl({
            type: 'model',
            task: selectedTask,
            hz,
            execution: { mode, request_threshold: threshold },
            // The follower eases onto the model's actions, which may start far from where it is.
            grace_s: 3,
        });

    return (
        <Flex direction='column' gap='size-200'>
            {tasks.length > 0 ? (
                <Picker
                    label='Task'
                    width='100%'
                    items={tasks.map((value) => ({ id: value }))}
                    selectedKey={selectedTask}
                    onSelectionChange={(key) => key !== null && setTask(String(key))}
                    isDisabled={isRunning}
                >
                    {(item) => <Item key={item.id}>{item.id}</Item>}
                </Picker>
            ) : (
                <TextArea label='Task' width='100%' value={task} onChange={setTask} isReadOnly={isRunning} />
            )}
            <Picker
                label='Execution'
                width='100%'
                items={EXECUTION_MODES}
                selectedKey={mode}
                onSelectionChange={(key) => key !== null && setMode(key as SchemaExecutionConfig['mode'])}
                isDisabled={isRunning}
            >
                {(item) => <Item key={item.key}>{item.label}</Item>}
            </Picker>
            <Slider
                label='Action rate'
                width='100%'
                minValue={HZ_RANGE.min}
                maxValue={HZ_RANGE.max}
                step={HZ_RANGE.step}
                value={hz}
                onChange={setHz}
                getValueLabel={(value) => `${value} Hz`}
                isDisabled={isRunning}
            />
            <Slider
                label='Request the next chunk at'
                width='100%'
                minValue={THRESHOLD_RANGE.min}
                maxValue={THRESHOLD_RANGE.max}
                step={THRESHOLD_RANGE.step}
                value={threshold}
                onChange={setThreshold}
                formatOptions={{ style: 'percent' }}
                isDisabled={isRunning}
            />
            {isRunning ? (
                <Button variant='secondary' isPending={isPending} onPress={() => void setControl(null)}>
                    Stop
                </Button>
            ) : (
                <Button isPending={isPending} isDisabled={!canRun} onPress={play}>
                    Play
                </Button>
            )}
        </Flex>
    );
};
