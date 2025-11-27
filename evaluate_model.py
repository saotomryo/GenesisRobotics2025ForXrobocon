"""
XROBOCON RL Model Evaluation Script
訓練済みモデルの定量的評価を行います
"""
import os
import numpy as np
from xrobocon.env import XRoboconEnv

def evaluate_model(model_path, num_episodes=10, render=False):
    """
    訓練済みモデルを評価
    
    Args:
        model_path: モデルファイルのパス
        num_episodes: 評価エピソード数
        render: 描画を有効にするか
    
    Returns:
        dict: 評価結果（成功率、平均報酬、平均ステップ数）
    """
    from stable_baselines3 import PPO
    
    render_mode = "human" if render else None
    env = XRoboconEnv(render_mode=render_mode)
    
    print(f"モデルを読み込み中: {model_path}")
    model = PPO.load(model_path)
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    distances = []
    
    print(f"\n{num_episodes}エピソードの評価を開始...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        min_dist = float('inf')
        
        while not done and steps < 1000:  # 最大1000ステップ
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # ターゲットまでの距離を計算
            robot_pos = env.robot.get_pos().cpu().numpy()
            target_pos = np.array(env.current_target['pos'])
            dist = np.linalg.norm(robot_pos[:2] - target_pos[:2])
            min_dist = min(min_dist, dist)
            
            # ターゲット到達判定（0.35m以内）
            if dist < 0.35:
                success_count += 1
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        distances.append(min_dist)
        
        success_str = "成功" if min_dist < 0.35 else "失敗"
        print(f"Episode {episode+1:2d}: 報酬={total_reward:7.2f}, ステップ={steps:4d}, 最小距離={min_dist:.3f}m, {success_str}")
    
    # 統計情報
    print("\n" + "="*60)
    print("評価結果")
    print("="*60)
    print(f"成功率:         {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"平均報酬:       {np.mean(total_rewards):7.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均ステップ数: {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):.1f}")
    print(f"平均最小距離:   {np.mean(distances):7.3f}m ± {np.std(distances):.3f}m")
    print("="*60)
    
    return {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(episode_lengths),
        'avg_min_distance': np.mean(distances)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='XROBOCON RL Model Evaluation')
    parser.add_argument('--model', type=str, default='xrobocon_ppo.zip', help='モデルファイルのパス')
    parser.add_argument('--episodes', type=int, default=20, help='評価エピソード数')
    parser.add_argument('--render', action='store_true', help='描画を有効にする')
    args = parser.parse_args()
    
    # Mac (MPS) 用の環境変数設定
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # モデルファイルの存在確認
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}")
        print("先に訓練を実行してください: python train_rl.py --train --steps 100000")
        exit(1)
    
    # 評価実行
    results = evaluate_model(args.model, num_episodes=args.episodes, render=args.render)
    
    # 評価基準の表示
    print("\n" + "="*60)
    print("評価基準")
    print("="*60)
    
    success_rate = results['success_rate']
    
    if success_rate >= 0.8:
        level = "優秀"
        comment = "実用レベルに達しています"
    elif success_rate >= 0.5:
        level = "良好"
        comment = "基本的な動作は習得済み。さらなる訓練で改善可能"
    elif success_rate >= 0.2:
        level = "要改善"
        comment = "基本動作は見られるが、訓練が不足しています"
    else:
        level = "不十分"
        comment = "訓練ステップ数を大幅に増やす必要があります"
    
    print(f"総合評価: {level}")
    print(f"コメント: {comment}")
    print("="*60)
