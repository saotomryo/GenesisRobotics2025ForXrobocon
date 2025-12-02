"""
XROBOCON RL Model Evaluation Script
訓練済みモデルの定量的評価を行います
"""
# 共通モジュールを最初にインポートして設定（MPS Fallback等）を適用
import xrobocon.common as common

import os
import numpy as np
import argparse
from xrobocon.env import XRoboconEnv
from xrobocon.step_env import XRoboconStepEnv
import xrobocon.common as common

def evaluate_model(model_path, num_episodes=10, render=False, robot_type='standard', env_type='flat'):
    """
    訓練済みモデルを評価
    
    Args:
        model_path: モデルファイルのパス
        num_episodes: 評価エピソード数
        render: 描画を有効にするか
        robot_type: ロボットタイプ ('standard', 'tristar')
        env_type: 環境タイプ ('flat', 'step')
    
    Returns:
        dict: 評価結果（成功率、平均報酬、平均ステップ数）
    """
    # 環境作成
    if env_type == 'step':
        env = XRoboconStepEnv(render_mode="human" if render else None, robot_type=robot_type)
    else:
        env = XRoboconEnv(render_mode="human" if render else None, robot_type=robot_type)
    
    # モデルロード
    model = common.load_trained_model(model_path, env)
    
    # 統計用データ
    stats = {
        'total': {'success': 0, 'reward': [], 'steps': [], 'dist': []},
        'scenarios': {} # シナリオごとの統計
    }
    
    print(f"\n{num_episodes}エピソードの評価を開始 ({env_type} environment)...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        min_dist = float('inf')
        
        # シナリオタイプ取得 (step_envの場合)
        scenario_type = info.get('scenario_type', 'default')
        if scenario_type not in stats['scenarios']:
            stats['scenarios'][scenario_type] = {'success': 0, 'reward': [], 'steps': [], 'dist': [], 'count': 0}
        
        while not done and steps < 500:  # 最大500ステップ
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
            
            # ターゲット到達判定（0.8m以内 - ロボットサイズを考慮）
            success = False
            if env_type == 'step':
                # 段差環境: 距離0.8m以内 かつ 高さ条件 (ターゲット高さ - 10cm)
                if dist < 0.8 and robot_pos[2] > target_pos[2] - 0.1:
                    success = True
            else:
                # 平地環境: 距離0.8m以内
                if dist < 0.8:
                    success = True
            
            if success:
                stats['total']['success'] += 1
                stats['scenarios'][scenario_type]['success'] += 1
                break
        
        # 統計更新
        stats['total']['reward'].append(total_reward)
        stats['total']['steps'].append(steps)
        stats['total']['dist'].append(min_dist)
        
        stats['scenarios'][scenario_type]['reward'].append(total_reward)
        stats['scenarios'][scenario_type]['steps'].append(steps)
        stats['scenarios'][scenario_type]['dist'].append(min_dist)
        stats['scenarios'][scenario_type]['count'] += 1
        
        print(f"Episode {episode+1:2d} [{scenario_type}]: 報酬={total_reward:7.2f}, ステップ={steps:4d}, 最小距離={min_dist:.3f}m, {'成功' if min_dist < 0.8 else '失敗'}")
    
    # 結果集計と表示
    print("\n" + "="*70)
    print("評価結果詳細")
    print("="*70)
    
    # 全体
    total_success_rate = stats['total']['success'] / num_episodes
    avg_reward = np.mean(stats['total']['reward'])
    std_reward = np.std(stats['total']['reward'])
    avg_steps = np.mean(stats['total']['steps'])
    avg_dist = np.mean(stats['total']['dist'])
    
    print(f"【全体】")
    print(f"  成功率:         {total_success_rate*100:.1f}% ({stats['total']['success']}/{num_episodes})")
    print(f"  平均報酬:       {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  平均ステップ数: {avg_steps:.1f}")
    print(f"  平均最小距離:   {avg_dist:.3f}m")
    
    # シナリオ別
    for s_type, s_stats in stats['scenarios'].items():
        count = s_stats['count']
        if count > 0:
            s_rate = s_stats['success'] / count
            s_reward = np.mean(s_stats['reward'])
            s_steps = np.mean(s_stats['steps'])
            s_dist = np.mean(s_stats['dist'])
            
            print(f"\n【{s_type}】 (n={count})")
            print(f"  成功率:         {s_rate*100:.1f}% ({s_stats['success']}/{count})")
            print(f"  平均報酬:       {s_reward:.2f}")
            print(f"  平均ステップ数: {s_steps:.1f}")
            print(f"  平均最小距離:   {s_dist:.3f}m")
            
    print("="*70)
    
    return {
        'success_rate': total_success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_steps': avg_steps,
        'avg_dist': avg_dist
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XROBOCON RL Model Evaluation')
    parser.add_argument('--model', type=str, default='xrobocon_ppo.zip', help='モデルファイルのパス')
    parser.add_argument('--episodes', type=int, default=20, help='評価エピソード数')
    parser.add_argument('--render', action='store_true', help='描画を有効にする')
    parser.add_argument('--robot', type=str, default='standard', help='ロボットタイプ (standard, tristar)')
    parser.add_argument('--env', type=str, default='flat', choices=['flat', 'step'], help='環境タイプ (flat, step)')
    args = parser.parse_args()
    
    # Mac (MPS) 用の環境変数設定
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # モデルファイルの存在確認
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}")
        print("先に訓練を実行してください")
        exit(1)
    
    # 評価実行
    results = evaluate_model(args.model, num_episodes=args.episodes, render=args.render, robot_type=args.robot, env_type=args.env)
    
    # 評価基準の表示
    print("\n" + "="*60)
    print("総合評価基準")
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
