import mujoco
import mujoco.viewer
import pathlib

import os

# Load the MuJoCo model from the XML file
model_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "constraints.xml"))
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Launch the viewer to simulate and visualize the scene
with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
