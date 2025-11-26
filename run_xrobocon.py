import genesis as gs
import torch
import numpy as np
import os
from xrobocon.field import XRoboconField
from xrobocon.robot import XRoboconRobot

class XRoboconSimulator:
    """XROBOCON シミュレーター"""
    
    def __init__(self):
        gs.init(backend=gs.gpu, precision='32')
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                res=(800, 600),
                max_FPS=60,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=0.005,
                gravity=(0.0, 0.0, -9.8),
            ),
            show_viewer=True,
        )
        
        # 地面
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        # フィールド作成
        self.field = XRoboconField()
        self.field_entities = self.field.build(self.scene)
        
        # ロボット作成 (地面に配置)
        # Ramp 1: End at (1.5, 0). Length 1.5. Direction Y+ (90 deg).
        # Start at (1.5, -1.5).
        # ロボットをその手前 (1.5, -2.0) に配置。向きはY軸正方向 (90度)。
        self.robot = XRoboconRobot(self.scene, pos=(1.5, -2.0, 0.1), euler=(0, 0, 90))
        
        self.scene.build()
        self.robot.post_build()
        
    def run(self):
        """シミュレーション実行"""
        print("Simulation started. Press Ctrl+C to exit.")
        
        step = 0
        try:
            while True:
                # Ramp 1 は Y軸正方向に登る。
                # すでにY軸正方向を向いているので、直進するだけ。
                self.robot.set_wheel_torques(20.0, 20.0)
                
                self.scene.step()
                step += 1
                
                if step % 100 == 0:
                    pos = self.robot.get_pos()
                    vel = self.robot.get_vel()
                    print(f"Step {step}: Pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Vel={vel}")
                
        except KeyboardInterrupt:
            print("Simulation stopped.")

if __name__ == "__main__":
    sim = XRoboconSimulator()
    sim.run()
