# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

> [!WARNING]
> MJWarp is in its Alpha stage, with many features still missing and limited testing so far.

MJWarp is written in [NVIDIA Warp](https://github.com/NVIDIA/warp), and is designed to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). Once MJWarp exits Alpha, it will be integrated back into both MJX and the upcoming Newton initiative.

MJWarp is maintained by [Google Deepmind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Installing for development

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

During early development, MJWarp is on the bleeding edge - you should install Warp nightly:

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
```

Then install MJWarp in editable mode for local development:

```
pip install -e .
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `XX passed in XX.XXs` at the end!

# Compatibility

The following features are implemented:

| Category          | Feature                            |
| ----------------- | ---------------------------------- |
| Dynamics          | Forward only                       |
| Transmission      | `JOINT`                            |
| Actuator Dynamics | `NONE`                             |
| Actuator Gain     | `FIXED`, `AFFINE`                  |
| Actuator Bias     | `NONE`, `AFFINE`                   |
| Geom              | `PLANE`, `SPHERE`, `CAPSULE`       |
| Constraint        | `LIMIT_JOINT`, `CONTACT_PYRAMIDAL` |
| Equality          | Not yet implemented                |
| Integrator        | `EULER`, `IMPLICITFAST`            |
| Cone              | `PYRAMIDAL`                        |
| Condim            | 3                                  |
| Solver            | `CG`, `NEWTON`                     |
| Fluid Model       | None                               |
| Tendons           | Not yet implemented.               |
| Sensors           | Not yet implemented.               |

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192 --event_trace=True
```

