"""
Object Interaction Example for Genesis Robot Simulator
Demonstrates robot-object interaction and manipulation
"""
import genesis as gs
import torch
import numpy as np


def main():
    """Object interaction demonstration"""
    print("Genesis Object Interaction Example")
    print("="*60)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu, precision='32')
    
    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -1.5, 1.8),
            camera_lookat=(0.4, 0.0, 0.5),
            camera_fov=35,
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
    
    # Add objects to interact with
    print("Adding objects to scene...")
    
    # Red box
    red_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.5, 0.0, 0.5),
            size=(0.05, 0.05, 0.05),
        ),
    )
    
    # Blue box
    blue_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.4, 0.15, 0.5),
            size=(0.06, 0.06, 0.06),
        ),
    )
    
    # Green sphere
    green_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.35, -0.15, 0.5),
            radius=0.03,
        ),
    )
    
    # Build scene
    scene.build()
    
    # Get robot info
    n_dofs = robot.n_dofs
    print(f"Robot loaded with {n_dofs} DOFs")
    print(f"Objects added: 2 boxes, 1 sphere\n")
    
    # Define poses
    home_pose = torch.zeros(n_dofs, device=gs.device)
    home_pose[1] = -0.5
    home_pose[3] = -2.0
    home_pose[5] = 1.5
    
    # Set initial pose
    robot.set_dofs_position(home_pose)
    
    print("Running interaction simulation...")
    print("The robot will reach toward the objects\n")
    
    # Simulation phases
    total_steps = 800
    
    for i in range(total_steps):
        current_pose = home_pose.clone()
        
        # Phase 1: Reach toward red box (0-200)
        if i < 200:
            # Extend arm forward
            progress = i / 200.0
            current_pose[1] = -0.5 + progress * 0.3  # Shoulder
            current_pose[3] = -2.0 + progress * 0.5  # Wrist
            
        # Phase 2: Move to blue box (200-400)
        elif i < 400:
            progress = (i - 200) / 200.0
            current_pose[0] = progress * 0.3  # Base rotation
            current_pose[1] = -0.2
            current_pose[3] = -1.5
            
        # Phase 3: Move to green sphere (400-600)
        elif i < 600:
            progress = (i - 400) / 200.0
            current_pose[0] = 0.3 - progress * 0.5  # Base rotation opposite
            current_pose[1] = -0.2
            current_pose[3] = -1.5
            
        # Phase 4: Return home (600-800)
        else:
            progress = (i - 600) / 200.0
            current_pose = home_pose * progress + current_pose * (1 - progress)
        
        # Gripper control: close when near objects, open otherwise
        if n_dofs > 7:
            if 150 < i < 250 or 350 < i < 450 or 550 < i < 650:
                current_pose[7:9] = 0.0  # Close gripper
            else:
                current_pose[7:9] = 0.04  # Open gripper
        
        # Update robot
        robot.set_dofs_position(current_pose)
        
        # Step simulation
        scene.step()
        
        # Print progress
        if i % 100 == 0:
            phase = i // 200 + 1
            phase_names = ["Reaching red box", "Moving to blue box", 
                          "Moving to green sphere", "Returning home"]
            if phase <= len(phase_names):
                print(f"Step {i}/{total_steps} - {phase_names[phase-1]}")
    
    print("\nInteraction example complete!")
    print("The robot demonstrated reaching toward multiple objects.")


if __name__ == "__main__":
    main()
