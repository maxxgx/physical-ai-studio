# Deploying Model Policies

| **Run model policy**               | **Model inference screen**                  |
|------------------------------------|---------------------------------------------|
| ![Run model policy][inference-run] | ![Model inference screen][inference-screen] |

[inference-run]: ./assets/07-inference-run.png
[inference-screen]: ./assets/07-inference-screen.png

Models can run inside Physical AI Studio or be deployed with [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai).
On the Models screen, click Run model for a trained policy, then select the inference backend and device. Studio currently supports PyTorch and OpenVINO on GPU or CPU.

When you start the model, Studio loads the environment used to record the dataset the model was trained on.
After the environment and model load, you will see a screen similar to the recording view. Pick the task the model should perform and click Play.

## Run a policy on a simulation from your own script

A simulation that Studio starts picks new ports every time and stops when Studio stops. To drive a simulated robot from your own script, start its simulation yourself:

1. Open the robot's start simulation dialog and pick the scene, and a seed under Advanced if you want one.
2. Open **Run it yourself** and copy the command.
3. Run the command in a terminal on the machine Studio runs on, with Studio's Python environment active (`source .venv/bin/activate` in `backend`). The simulation runs until you stop it with Ctrl+C.

Studio connects to it as it is: the robot shows `Running · external`, with its 3D viewer and cameras.

To run a trained policy on it from a script:

1. On the Models screen, open the model and click **Runtime export** on its OpenVINO format.
2. Pick an environment with that simulated robot and a device, and download the zip.
3. Unzip it and run the policy from that folder:

```bash
physicalai run --config runtime.yaml
```

Limits:

- This works with a native install only. A simulation accepts robot connections from its own machine only, so Studio in Docker can't reach a simulation started on the host, and a script on the host can't reach one that Studio started in its container.
- One Studio session or one script drives the robot at a time. Stop one before you start the other.
- Keep the port the robot is configured with. The command uses the robot's name and address, and Studio's cameras and `runtime.yaml` point at that address. If you change the port in the command, change the robot's address in Studio to match.

## Next

- For training, see [Training Policies](./06-training-policies.md).
