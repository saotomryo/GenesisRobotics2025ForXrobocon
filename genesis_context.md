# System Context: Genesis (Genesis-Embodied-AI) Programming Guide

**Instruction to Agent:**
You are an expert in using **Genesis**, a Universal Physics Platform for Embodied AI. Genesis is a PyTorch-based, differentiable, and generative physics engine.
Since this library is new, **DO NOT** rely on your pre-trained knowledge of other simulators (like Isaac Gym or MuJoCo) unless the syntax is identical. **Strictly follow the API structures and examples provided below.**

## 1. Core Architecture & Initialization
Genesis uses a unified backend (Taichi/CUDA). Always initialize `genesis` first.

```python
import genesis as gs
import torch

# 1. Initialization
# backend: gs.gpu, gs.cpu, gs.vulkan
# precision: '32' or '64'
gs.init(backend=gs.gpu, precision='32')

# 2. Scene Creation
scene = gs.Scene(
    # Viewer (GUI) settings
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
    ),
    # Rigid Body Solver settings
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -9.8),
    ),
    # MPM Solver (for soft bodies, sand, etc.) - Optional
    mpm_options=gs.options.MPMOptions(
        dt=1e-3, # MPM needs smaller dt
        lower_bound=(-1.0, -1.0, -0.1),
        upper_bound=(1.0, 1.0, 1.0),
    ),
    # SPH Solver (for liquids) - Optional
    sph_options=gs.options.SPHOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
        particle_size=0.01,
    )
)

# Plane (Ground)
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# Robot from MJCF or URDF
franka = scene.add_entity(
    gs.morphs.MJCF(file='path/to/panda.xml'), # or gs.morphs.URDF(...)
    pos=(0.0, 0.0, 0.0),
    euler=(0, 0, 0), # Rotation in degrees
)

# Rigid Primitive (Box, Sphere, etc.)
box = scene.add_entity(
    gs.morphs.Box(
        pos=(0.5, 0.0, 0.5),
        size=(0.1, 0.1, 0.1),
    )
)

# Elastic object (Soft body)
soft_box = scene.add_entity(
    gs.morphs.Elastic(
        file='path/to/mesh.obj',
        pos=(0.5, 0.5, 0.5),
        scale=0.1,
    ),
    material=gs.materials.MPM.Elastic(
        E=1e6, # Young's modulus
        nu=0.2, # Poisson's ratio
        rho=1000.0, # Density
    )
)

# Liquid (SPH or MPM)
water = scene.add_entity(
    gs.morphs.Liquid(
        pos=(0.0, 0.0, 1.0),
        size=(0.1, 0.1, 0.1), # Emitter size
    ),
    surface=gs.surfaces.Rough(
        color=(0.0, 0.0, 1.0),
        vis_mode='particle',
    )
)

# Get number of DOFs
n_dofs = franka.n_dofs

# Set Position Control (PD Control is internal default for many settings)
franka.set_dofs_position(
    position=torch.tensor([0.0, -0.7, 0.0, -2.0, 0.0, 1.5, 0.78]), # Shape must match n_dofs
    dofs_idx_local=[0, 1, 2, 3, 4, 5, 6] # Optional: specific joints
)

# Set Velocity
franka.set_dofs_velocity(velocity=torch.tensor([...]))

# Set Force/Torque directly
franka.set_dofs_force(force=torch.tensor([...]))

# Inverse Kinematics (IK) - Basic usage
target_pos = (0.5, 0.0, 0.5)
target_quat = (0, 1, 0, 0)
q_sol = franka.inverse_kinematics(
    link=franka.get_link('hand'), # Specify end-effector link
    pos=target_pos,
    quat=target_quat,
)

import genesis as gs
import torch

def main():
    gs.init(backend=gs.gpu, precision='32')

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
    )

    # Add entities
    plane = scene.add_entity(gs.morphs.Plane())
    
    # NOTE: Ensure the file path exists or use a placeholder
    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    )

    scene.build()

    # Move to initial pose
    motors_dof = franka.n_dofs
    # Example pose
    initial_pose = torch.zeros(motors_dof, device=gs.device) 
    initial_pose[1] = -0.5 
    initial_pose[3] = -2.0
    
    franka.set_dofs_position(initial_pose)

    # Run simulation
    for i in range(500):
        # Dynamic control example: Sine wave on first joint
        # initial_pose[0] = torch.sin(torch.tensor(i * 0.01))
        # franka.set_dofs_position(initial_pose)
        
        scene.step()

if __name__ == "__main__":
    main()