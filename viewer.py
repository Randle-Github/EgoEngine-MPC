import sys
import mujoco
import mujoco.viewer

if len(sys.argv) < 2:
    print("Usage: python run_mujoco.py <path_to_xml>")
    sys.exit(1)

xml_path = sys.argv[1]

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
