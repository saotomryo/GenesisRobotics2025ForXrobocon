import genesis as gs
import torch
import numpy as np
import os

class XRoboconRobot:
    """XROBOCON ロボットクラス"""
    
    def __init__(self, scene, pos=(0, 0, 0.1), euler=(0, 0, 0)):
        self.scene = scene
        
        # アセットパス
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        robot_path = os.path.join(assets_dir, 'robot.xml')
        
        # ロボットエンティティを追加
        self.entity = self.scene.add_entity(
            gs.morphs.MJCF(
                file=robot_path,
                pos=pos,
                euler=euler,
            )
        )
        
        self.n_dofs = 0 # ビルド後に更新
        
    def post_build(self):
        """シーンビルド後の初期化"""
        self.n_dofs = self.entity.n_dofs
        print(f"Robot initialized with {self.n_dofs} DOFs")
        
    def set_wheel_torques(self, left, right):
        """左右のホイールにトルクを適用"""
        if self.n_dofs < 2:
            return
            
        forces = torch.zeros(self.n_dofs, device=gs.device)
        # 最後の2つがホイール用モーターと仮定
        forces[-2] = float(left)
        forces[-1] = float(right)
        self.entity.control_dofs_force(forces)
        
    def set_pose(self, pos, euler_deg):
        """位置と姿勢(オイラー角:度)を設定"""
        # 位置設定
        self.entity.set_pos(pos)
        
        # オイラー角(度) -> クォータニオン変換
        # Genesis/MuJoCo uses [w, x, y, z]
        roll = np.radians(euler_deg[0])
        pitch = np.radians(euler_deg[1])
        yaw = np.radians(euler_deg[2])
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        self.entity.set_quat([w, x, y, z])
        
        # 速度リセット
        self.entity.set_dofs_velocity(torch.zeros(self.n_dofs, device=gs.device))

    def get_pos(self):
        """ロボットの位置を取得"""
        return self.entity.get_pos()
        
    def get_euler(self):
        """ロボットのオイラー角(度)を取得"""
        quat = self.entity.get_quat().cpu().numpy() # [w, x, y, z]
        w, x, y, z = quat
        
        # クォータニオン -> オイラー角変換
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return np.degrees([roll_x, pitch_y, yaw_z])
        
    def get_vel(self):
        """ロボットの線形速度を取得"""
        # 浮遊ベース(freejoint)の場合、DOF速度の最初3つが線形速度
        if self.n_dofs >= 6:
            return self.entity.get_dofs_velocity()[:3]
        return self.entity.get_vel() # フォールバック
        
    def get_ang_vel(self):
        """ロボットの角速度を取得"""
        # 浮遊ベース(freejoint)の場合、DOF速度の次の3つが角速度
        if self.n_dofs >= 6:
            return self.entity.get_dofs_velocity()[3:6]
        return torch.zeros(3, device=gs.device) # フォールバック
        
    def get_camera_frame(self):
        """カメラ画像を取得 (未実装: Genesisのレンダリングパイプラインが必要)"""
        # Genesisでのカメラ画像取得はシーンのレンダラー経由で行う必要がある
        # ここではプレースホルダー
        pass
