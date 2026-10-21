import { FormPreviewLayout } from '../../components/form-preview-layout';
import { Preview } from '../../features/robots/environment-form/preview';
import {
    EnvironmentFormProvider,
    EnvironmentFormState,
    RobotConfiguration,
} from '../../features/robots/environment-form/provider';
import { UpdateEnvironmentForm } from '../../features/robots/environment-form/update-form';
import { useEnvironment } from '../../features/robots/use-environment';
import { SimulationCamerasPrompt } from '../../features/simulation/simulation-cameras-prompt';

export const Edit = () => {
    const environment = useEnvironment();

    const environmentForm: EnvironmentFormState = {
        name: environment.name,
        cameras: environment.cameras?.map(({ id, name }) => ({ camera_id: id!, name: name! })) ?? [],
        robots:
            environment.robots?.map((robot): RobotConfiguration => {
                return {
                    robot_id: robot.robot.id,
                    teleoperator:
                        robot.tele_operator.type === 'robot'
                            ? {
                                  type: 'robot',
                                  robot_id: robot.tele_operator.robot_id,
                              }
                            : { type: 'none' },
                };
            }) ?? [],
    };

    return (
        // Keyed by the cameras: the form state is built once, and cameras added from the
        // simulation prompt need new preview panels and a session that opens them.
        <EnvironmentFormProvider
            key={environmentForm.cameras.map(({ camera_id }) => camera_id).join()}
            environment={environmentForm}
        >
            <FormPreviewLayout
                form={
                    <>
                        <SimulationCamerasPrompt environment={environment} />
                        <UpdateEnvironmentForm />
                    </>
                }
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50' }}
            />
        </EnvironmentFormProvider>
    );
};
