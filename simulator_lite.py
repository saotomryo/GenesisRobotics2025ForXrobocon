"""
軽量版シミュレーター - リソース制約環境向け
オブジェクトなし、低解像度でパフォーマンス最適化
"""
import genesis as gs
import torch
import numpy as np

# 軽量設定
CAMERA_CONFIG = {
    'pos': (2.5, -1.0, 1.5),
    'lookat': (0.0, 0.0, 0.5),
    'fov': 40,
    'resolution': (640, 480),  # さらに低解像度
    'max_fps': 30,
}

PHYSICS_CONFIG = {
    'dt': 0.02,  # タイムステップを大きくして計算量削減
    'gravity': (0.0, 0.0, -9.8),
}

HOME_POSE = torch.tensor([
    0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.78,
])


def main():
    """軽量版メイン"""
    print("Genesis 軽量版ロボットシミュレーター")
    print("="*60)
    
    # 初期化
    gs.init(backend=gs.gpu, precision='32')
    
    # シーン作成
    scene = gs.Scene(
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
    
    # 地面のみ追加
    plane = scene.add_entity(gs.morphs.Plane())
    
    # ロボット追加
    print("ロボットを読み込み中...")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/franka_emika_panda/panda.xml',
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
        )
    )
    
    # ビルド
    scene.build()
    
    n_dofs = robot.n_dofs
    print(f"ロボット自由度: {n_dofs}")
    
    # 初期姿勢
    current_pose = HOME_POSE.clone().to(gs.device)
    if len(current_pose) < n_dofs:
        padding = torch.zeros(n_dofs - len(current_pose), device=gs.device)
        current_pose = torch.cat([current_pose, padding])
    
    robot.set_dofs_position(current_pose)
    
    print("\n軽量デモモード実行中...")
    print("ビジュアライザーが開きます。Ctrl+Cで終了。\n")
    
    step = 0
    gripper_open = True
    
    try:
        while True:
            # ゆっくりした動作
            if step % 100 == 0:
                angle = np.sin(step * 0.005) * 0.3  # より遅く、小さく
                current_pose[0] = angle
                robot.set_dofs_position(current_pose)
            
            # グリッパー制御（頻度を下げる）
            if step % 500 == 0 and step > 0:
                gripper_open = not gripper_open
                if n_dofs > 7:
                    gripper_pos = 0.04 if gripper_open else 0.0
                    current_pose[7:9] = gripper_pos
                    robot.set_dofs_position(current_pose)
                    print(f"グリッパー: {'開' if gripper_open else '閉'}")
            
            scene.step()
            
            if step % 1000 == 0:
                print(f"ステップ {step}")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n\n停止しました")
    
    print("完了！")


if __name__ == "__main__":
    main()
