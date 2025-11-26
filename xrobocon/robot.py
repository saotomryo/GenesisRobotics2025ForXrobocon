import genesis as gs
import torch
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
        forces[-2] = left
        forces[-1] = right
        self.entity.control_dofs_force(forces)
        
    def get_pos(self):
        """ロボットの位置を取得"""
        return self.entity.get_pos()
        
    def get_vel(self):
        """ロボットの速度を取得"""
        return self.entity.get_vel()
        
    def get_camera_frame(self):
        """カメラ画像を取得 (未実装: Genesisのレンダリングパイプラインが必要)"""
        # Genesisでのカメラ画像取得はシーンのレンダラー経由で行う必要がある
        # ここではプレースホルダー
        pass
