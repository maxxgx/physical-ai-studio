import { useState } from 'react';

import { Button, ButtonGroup, Flex, Grid, View } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { useProjectId } from '../../features/projects/use-project';
import { JointControls } from '../../features/robots/controller/joint-controls';
import { RobotViewer, UnavailableRobotViewer } from '../../features/robots/controller/robot-viewer';
import { useCatalogIdentifyMutation } from '../../features/robots/robot-catalog.hooks';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../features/robots/robot-types';
import { useRobot } from '../../features/robots/use-robot';
import { robotSimulationStatus, useRobotSimulationLaunch } from '../../features/simulation/robot-simulation';
import { useSimulationDialog } from '../../features/simulation/simulation-dialog';
import { SimulationNotRunning } from '../../features/simulation/simulation-not-running';
import { useSimulation } from '../../features/simulation/use-simulation';

export const Robot = () => {
    const robot = useRobot();
    return isUnavailableRobot(robot) ? (
        <UnavailableRobotViewer robotType={robot.type} />
    ) : (
        <AvailableRobot robot={robot} />
    );
};

/** Whether a simulated robot's simulation is down, so the page shows how to start it instead. */
const useIsSimulationDown = (robot: AvailableSchemaRobot) => {
    const { project_id } = useProjectId();
    const launch = useRobotSimulationLaunch(robot.type);
    const { data: simulation } = useSimulation();
    const { data: onlineRobots } = $api.useQuery('get', '/api/projects/{project_id}/robots/online', {
        params: { path: { project_id } },
    });
    const connectionStatus = onlineRobots?.find(({ id }) => id === robot.id)?.connection_status ?? 'unknown';
    const status = robotSimulationStatus(robot.id, simulation, connectionStatus);

    return launch !== null && (status === 'offline' || status === 'starting' || status === 'stopping');
};

const AvailableRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const isSimulationDown = useIsSimulationDown(robot);
    const simulationDialog = useSimulationDialog(robot);

    return (
        <>
            {isSimulationDown ? (
                <SimulationNotRunning robot={robot} onStart={simulationDialog.open} />
            ) : (
                <ConnectableRobot robot={robot} />
            )}
            {simulationDialog.dialog}
        </>
    );
};

const ConnectableRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const identifyMutation = useCatalogIdentifyMutation();

    const onIdentify = identifyMutation.isPending
        ? undefined
        : () => {
              identifyMutation.mutate({
                  params: { path: { robot_type: robot.type } },
                  body: robot.payload,
              });
          };

    const [isConnected, setIsConnected] = useState(false);

    return (
        <View height='100%' minHeight='0'>
            <RobotModelsProvider>
                <Grid
                    areas={['actions', 'robot-viewer', 'controls']}
                    rows={['auto', '1fr', 'min-content']}
                    height='100%'
                    maxHeight={'100vh'}
                    maxWidth='100%'
                    minHeight={0}
                    minWidth={0}
                >
                    <View
                        gridColumn='1/-1'
                        gridRow='1/-1'
                        overflow='auto'
                        zIndex={0}
                        minHeight={0}
                        UNSAFE_style={
                            isConnected
                                ? undefined
                                : {
                                      filter: 'grayscale(0.8)',
                                      opacity: 0.5,
                                  }
                        }
                    >
                        <RobotViewer robot={robot} />
                    </View>
                    <View gridArea='actions' zIndex={1} margin='size-400'>
                        <Flex justifyContent={'end'}>
                            <ButtonGroup>
                                <Button variant='secondary' onPress={onIdentify}>
                                    Identify
                                </Button>
                            </ButtonGroup>
                        </Flex>
                    </View>
                    <JointControls isConnected={isConnected} setIsConnected={setIsConnected} />
                </Grid>
            </RobotModelsProvider>
        </View>
    );
};
