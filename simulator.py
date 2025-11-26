"""
Genesis ロボットシミュレーター - メインアプリケーション
Franka Emika Pandaロボットアームを使用したインタラクティブロボットシミュレーター
"""
import genesis as gs
import torch
import numpy as np
from config import (
    CAMERA_CONFIG, PHYSICS_CONFIG, ROBOT_CONFIG,
    HOME_POSE, CONTROL_CONFIG, SCENE_OBJECTS
)


class RobotSimulator:
    """Genesisロボット環境のメインシミュレータークラス"""
    
    def __init__(self):
        """シミュレーターを初期化"""
        # Genesisを初期化
        gs.init(backend=gs.gpu, precision='32')
        
        # シーンを作成
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=CAMERA_CONFIG['pos'],
                camera_lookat=CAMERA_CONFIG['lookat'],
                camera_fov=CAMERA_CONFIG['fov'],
                res=CAMERA_CONFIG['resolution'],
                max_FPS=CAMERA_CONFIG['max_fps'],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=PHYSICS_CONFIG['dt'],
                gravity=PHYSICS_CONFIG['gravity'],
            ),
            show_viewer=True,  # ビューアを明示的に表示
        )
        
        # 地面を追加
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        
        # ロボットを追加
        print(f"ロボットを読み込み中: {ROBOT_CONFIG['model_path']}")
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=ROBOT_CONFIG['model_path'],
                pos=ROBOT_CONFIG['initial_position'],
                euler=ROBOT_CONFIG['initial_euler'],
            )
        )
        
        # シーンオブジェクトを追加
        self.objects = []
        self._add_scene_objects()
        
        # シーンをビルド
        self.scene.build()
        
        # ロボットの状態を初期化
        self.n_dofs = self.robot.n_dofs
        print(f"ロボットの自由度: {self.n_dofs}")
        
        # 初期姿勢を設定
        self.current_pose = HOME_POSE.clone().to(gs.device)
        if len(self.current_pose) < self.n_dofs:
            # 必要に応じてグリッパー関節用にゼロでパディング
            padding = torch.zeros(self.n_dofs - len(self.current_pose), device=gs.device)
            self.current_pose = torch.cat([self.current_pose, padding])
        
        self.robot.set_dofs_position(self.current_pose)
        
        # 制御状態
        self.selected_joint = 0
        self.gripper_open = True
        
    def _add_scene_objects(self):
        """設定に基づいてシーンにオブジェクトを追加"""
        for obj_config in SCENE_OBJECTS:
            if obj_config['type'] == 'box':
                obj = self.scene.add_entity(
                    gs.morphs.Box(
                        pos=obj_config['pos'],
                        size=obj_config['size'],
                    ),
                )
            elif obj_config['type'] == 'sphere':
                obj = self.scene.add_entity(
                    gs.morphs.Sphere(
                        pos=obj_config['pos'],
                        radius=obj_config['radius'],
                    ),
                )
            self.objects.append(obj)
    
    def reset_to_home(self):
        """ロボットをホームポジションにリセット"""
        self.current_pose[:len(HOME_POSE)] = HOME_POSE.to(gs.device)
        self.robot.set_dofs_position(self.current_pose)
        print("ホームポジションにリセットしました")
    
    def adjust_joint(self, joint_idx, delta):
        """指定した関節をdeltaラジアン分調整"""
        if 0 <= joint_idx < min(7, self.n_dofs):  # アーム関節のみ制御、グリッパーは除外
            self.current_pose[joint_idx] += delta
            self.robot.set_dofs_position(self.current_pose)
            print(f"関節 {joint_idx}: {self.current_pose[joint_idx]:.3f} rad")
    
    def toggle_gripper(self):
        """グリッパーの開閉を切り替え"""
        if self.n_dofs > 7:  # グリッパー関節がある場合
            self.gripper_open = not self.gripper_open
            gripper_pos = CONTROL_CONFIG['gripper_open'] if self.gripper_open else CONTROL_CONFIG['gripper_closed']
            # 両方のグリッパー指を設定
            self.current_pose[7:9] = gripper_pos
            self.robot.set_dofs_position(self.current_pose)
            print(f"グリッパー: {'開' if self.gripper_open else '閉'}")
    
    def print_controls(self):
        """操作方法を表示"""
        print("\n" + "="*60)
        print("GENESIS ロボットシミュレーター - 操作方法")
        print("="*60)
        print("数字キー 1-7: 制御する関節を選択")
        print("+/=: 選択した関節の角度を増加")
        print("-/_: 選択した関節の角度を減少")
        print("R: ホームポジションにリセット")
        print("G: グリッパーの開閉")
        print("ESC: 終了")
        print("="*60)
        print(f"現在制御中: 関節 {self.selected_joint}")
        print("="*60 + "\n")
    
    def run(self, max_steps=None):
        """
        シミュレーションを実行
        
        Args:
            max_steps: 最大ステップ数（Noneで無限）
        """
        self.print_controls()
        
        step_count = 0
        try:
            while max_steps is None or step_count < max_steps:
                # シミュレーションをステップ実行
                self.scene.step()
                step_count += 1
                
                # 注意: キーボード入力処理には追加の統合が必要
                # Genesisビューアまたは外部入力ライブラリとの連携
                # 現在は可視化モードでシミュレーションを実行
                
        except KeyboardInterrupt:
            print("\nユーザーによってシミュレーションが停止されました")
    
    def run_demo(self, steps=1000):
        """
        自動動作のデモを実行
        
        Args:
            steps: シミュレーションステップ数
        """
        print("\n自動動作デモモードを実行中...")
        print("ビジュアライザーウィンドウが開きます。閉じるまでシミュレーションが続きます。")
        print("注意: リソース不足の場合は、config.pyで解像度を下げてください。\n")
        
        step = 0
        try:
            while True:
                # 関節0に単純な正弦波動作を作成（更新頻度を下げる）
                if step % 50 == 0:  # 100から50に変更してスムーズに
                    angle = np.sin(step * 0.01) * 0.5
                    self.current_pose[0] = angle
                    self.robot.set_dofs_position(self.current_pose)
                
                # 定期的にグリッパーを切り替え
                if step % 300 == 0 and step > 0:  # 200から300に変更して頻度を下げる
                    self.toggle_gripper()
                
                self.scene.step()
                
                # 出力頻度を下げる
                if step % 500 == 0:  # 100から500に変更
                    print(f"ステップ {step} - FPS変動は正常です")
                
                step += 1
                
        except KeyboardInterrupt:
            print("\n\nユーザーによって停止されました")
        
        print("デモ完了！")


def main():
    """メインエントリーポイント"""
    print("Genesis ロボットシミュレーターを初期化中...")
    
    # シミュレーターを作成
    sim = RobotSimulator()
    
    # デモモードを実行（ビジュアライザーが開きます）
    # ウィンドウを閉じるか、Ctrl+Cで終了できます
    sim.run_demo()
    
    # または、可視化モードで実行:
    # sim.run()


if __name__ == "__main__":
    main()
