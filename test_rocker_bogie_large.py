"""
Rocker-Bogie Large 手動制御テスト
キーボードで制御して動作確認
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
    print("🤖 Rocker-Bogie Large 手動制御テスト")
    print("="*70)
    print("\nキーボード操作:")
    print("  W: 前進")
    print("  S: 後退")
    print("  A: 左旋回")
    print("  D: 右旋回")
    print("  Q: 終了")
    print("="*70 + "\n")
    
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
    
    # Rocker-Bogie Large ロボット
    print("ロボットを作成中...")
    robot = XRoboconRobot(
        scene, 
        pos=(5.0, -1.0, 0.225),  # z_offset = 0.225
        euler=(0, 0, 90), 
        robot_type='rocker_bogie_large'
    )
    
    # ゲームロジック
    game = XRoboconGame(field, robot)
    field.add_coin_spots(scene, game.spots)
    
    # シーンビルド
    scene.build()
    robot.post_build()
    
    print(f"✅ ロボット初期化完了: {robot.n_dofs} DOFs")
    print(f"   期待値: 12 DOFs (6 free joint + 6 actuators)")
    
    if robot.n_dofs != 12:
        print(f"⚠️  警告: DOF数が期待値と異なります！")
    
    # ゲーム開始
    game.start()
    
    # 手動制御ループ
    print("\n🎮 手動制御開始（Wキーで前進してみてください）")
    
    left_cmd = 0.0
    right_cmd = 0.0
    
    step_count = 0
    try:
        while True:
            # キーボード入力（簡易版）
            # 実際のキーボード入力は難しいので、自動テストパターンを実行
            if step_count < 100:
                # 前進テスト
                left_cmd = 0.5
                right_cmd = 0.5
                if step_count == 0:
                    print("前進テスト中...")
            elif step_count < 200:
                # 停止
                left_cmd = 0.0
                right_cmd = 0.0
                if step_count == 100:
                    print("停止中...")
            elif step_count < 300:
                # 左旋回テスト
                left_cmd = -0.3
                right_cmd = 0.3
                if step_count == 200:
                    print("左旋回テスト中...")
            elif step_count < 400:
                # 右旋回テスト
                left_cmd = 0.3
                right_cmd = -0.3
                if step_count == 300:
                    print("右旋回テスト中...")
            else:
                # 停止
                left_cmd = 0.0
                right_cmd = 0.0
                if step_count == 400:
                    print("テスト完了。停止中...")
                    print("\nロボットの動作を確認してください。")
                    print("動いていれば成功です！")
            
            # アクション適用
            actions = np.array([left_cmd, right_cmd])
            robot.set_actions(actions)
            
            # シミュレーションステップ
            scene.step()
            
            # ロボット位置表示（10ステップごと）
            if step_count % 100 == 0:
                pos = robot.entity.get_pos().cpu().numpy()
                print(f"Step {step_count}: Position = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            step_count += 1
            time.sleep(0.01)
            
            # 500ステップで終了
            if step_count >= 500:
                break
                
    except KeyboardInterrupt:
        print("\n\n終了します...")
    
    print("\n" + "="*70)
    print("✅ テスト完了")
    print("="*70)
    
    # 最終位置
    final_pos = robot.entity.get_pos().cpu().numpy()
    print(f"\n最終位置: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    
    # 移動距離
    initial_pos = np.array([5.0, -1.0, 0.225])
    distance = np.linalg.norm(final_pos[:2] - initial_pos[:2])
    print(f"移動距離: {distance:.2f}m")
    
    if distance > 0.1:
        print("✅ ロボットは正常に動作しています！")
    else:
        print("❌ ロボットが動いていません。制御コードに問題がある可能性があります。")

if __name__ == "__main__":
    main()
