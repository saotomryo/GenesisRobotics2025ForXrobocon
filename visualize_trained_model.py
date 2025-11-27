"""
訓練済みモデルの動作を可視化し、停滞時点のスクリーンショットを撮影
"""
import os
import numpy as np
import genesis as gs
from xrobocon.env import XRoboconEnv
from stable_baselines3 import PPO

def visualize_trained_model():
    """訓練済みモデルの動作を可視化"""
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # 環境作成（描画あり）
    # カメラを訓練エリア（X=5.0付近）に向ける
    env = XRoboconEnv(render_mode="human")
    
    # カメラ位置を訓練エリアに調整
    # 訓練エリアは(5.0, 0.0, 0.0)付近
    env.scene.viewer.set_camera_pose(pos=np.array([8.0, -3.0, 2.5]), lookat=np.array([5.0, 0.0, 0.0]))
    
    # モデル読み込み
    print("モデルを読み込み中...")
    model = PPO.load("xrobocon_ppo", env=env)
    
    print("\n" + "="*70)
    print("訓練済みモデルの動作を可視化")
    print("="*70)
    print("ロボットが目標に最も近づいた時点で一時停止します")
    print("Escキーでウィンドウを閉じると次のエピソードに進みます")
    print("="*70 + "\n")
    
    num_episodes = 3
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        print(f"\nエピソード {episode + 1}/{num_episodes}")
        print(f"  初期位置: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
        print(f"  目標位置: ({env.current_target['pos'][0]:.2f}, {env.current_target['pos'][1]:.2f}, {env.current_target['pos'][2]:.2f})")
        
        min_dist = 999
        min_dist_step = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # 現在の距離を計算
            robot_pos = env.robot.get_pos().cpu().numpy()
            target_pos = np.array(env.current_target['pos'])
            dist = np.linalg.norm(robot_pos[:2] - target_pos[:2])
            
            if dist < min_dist:
                min_dist = dist
                min_dist_step = step
            
            # 最小距離から離れ始めたら一時停止
            if step > min_dist_step + 20 and step > 50:
                print(f"  最小距離到達: Step {min_dist_step}, 距離 {min_dist:.3f}m")
                print(f"  現在のステップ: {step}")
                print(f"  ロボット位置: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
                print(f"  目標からの距離: {dist:.3f}m")
                print(f"\n  ⚠️  ロボットが目標から離れ始めました")
                print(f"  ビューアで状況を確認してください")
                print(f"  確認したらEnterキーを押して次のエピソードに進みます...")
                
                # ユーザーの入力を待つ
                input()
                
                break
            
            if terminated or truncated:
                print(f"  エピソード終了: Step {step}")
                break
        
        print(f"  最終的な最小距離: {min_dist:.3f}m")
        print(f"  成功判定（0.3m以内）: {'成功' if min_dist < 0.3 else '失敗'}")
    
    print("\n" + "="*70)
    print("可視化完了")
    print("="*70)

if __name__ == "__main__":
    visualize_trained_model()
