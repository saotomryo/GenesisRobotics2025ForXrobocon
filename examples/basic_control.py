"""
Basic Control Example for Genesis Robot Simulator
Demonstrates simple robot loading and joint control
"""
import genesis as gs
import torch
import numpy as np


def main():
    """Basic control demonstration"""
    print("Genesis Basic Control Example")
    print("="*60)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu, precision='32')
    
    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -1.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.8),
        ),
    )
    
    # Add ground
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Add robot
    print("Loading Franka Panda robot...")
    robot = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    )
    
    # Build scene
    scene.build()
    
    # Get robot info
    n_dofs = robot.n_dofs
    print(f"Robot loaded successfully with {n_dofs} DOFs")
    
    # Define home position
    home_pose = torch.zeros(n_dofs, device=gs.device)
    home_pose[1] = -0.5
    home_pose[3] = -2.0
    home_pose[5] = 1.5
    
    # Set initial pose
    robot.set_dofs_position(home_pose)
    
    print("\nRunning simulation with automated joint movements...")
    print("Watch the robot move through different poses\n")
    
    # Run simulation with different movements
    for i in range(600):
        # Phase 1: Move base joint (0-200 steps)
        if i < 200:
            angle = np.sin(i * 0.02) * 0.8
            home_pose[0] = angle
        
        # Phase 2: Move shoulder joint (200-400 steps)
        elif i < 400:
            angle = -0.5 + np.sin((i - 200) * 0.02) * 0.5
            home_pose[1] = angle
        
        # Phase 3: Move elbow joint (400-600 steps)
        else:
            angle = np.sin((i - 400) * 0.02) * 0.7
            home_pose[2] = angle
        
        # Update robot position
        robot.set_dofs_position(home_pose)
        
        # Step simulation
        scene.step()
        
        # Print progress
        if i % 100 == 0:
            print(f"Step {i}/600 - Phase {i//200 + 1}/3")
    
    print("\nExample complete!")
    print("The robot demonstrated movement of base, shoulder, and elbow joints.")


if __name__ == "__main__":
    main()
