"""
元のRocker-Bogie テスト
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import xrobocon.common as common
import genesis as gs
import numpy as np
import time
from xrobocon.field import XRoboconField
from xrobocon.robot import XRoboconRobot
from xrobocon.game import XRoboconGame

def main():
    print("\n" + "="*70)
    print("🤖 元のRocker-Bogie テスト")
    print("="*70)
    
    # Genesis初期化
    common.setup_genesis()
    
    # シーン作成
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.8),
        ),
        show_viewer=True,
    )
    
    # 地面
    plane = scene.add_entity(gs.morphs.Plane())
    
    # フィールド
    field = XRoboconField()
    field.build(scene)
    
    # 元のRocker-Bogie
    print("ロボットを作成中...")
    robot = XRoboconRobot(
        scene, 
        pos=(5.0, -1.0, 0.15),
        euler=(0, 0, 90), 
        robot_type='rocker_bogie'
    )
    
    # ゲームロジック
    game = XRoboconGame(field, robot)
    field.add_coin_spots(scene, game.spots)
    
    # シーンビルド
    scene.build()
    robot.post_build()
    
    print(f"✅ ロボット初期化完了: {robot.n_dofs} DOFs")
    
    # ゲーム開始
    game.start()
    
    # 手動制御ループ
    print("\n🎮 前進テスト開始")
    
    left_cmd = 0.5
    right_cmd = 0.5
    
    step_count = 0
    try:
        while step_count < 500:
            # アクション適用
            actions = np.array([left_cmd, right_cmd])
            robot.set_actions(actions)
            
            # シミュレーションステップ
            scene.step()
            
            # ロボット位置表示（100ステップごと）
            if step_count % 100 == 0:
                pos = robot.entity.get_pos().cpu().numpy()
                print(f"Step {step_count}: Position = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            step_count += 1
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\n終了します...")
    
    print("\n" + "="*70)
    print("✅ テスト完了")
    print("="*70)
    
    # 最終位置
    final_pos = robot.entity.get_pos().cpu().numpy()
    print(f"\n最終位置: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    
    # 移動距離
    initial_pos = np.array([5.0, -1.0, 0.15])
    distance = np.linalg.norm(final_pos[:2] - initial_pos[:2])
    print(f"移動距離: {distance:.2f}m")
    
    if distance > 0.1:
        print("✅ ロボットは正常に動作しています！")
    else:
        print("❌ ロボットが動いていません。")

if __name__ == "__main__":
    main()
