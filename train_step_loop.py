import os
import argparse
import time
from stable_baselines3 import PPO
from xrobocon.step_env import XRoboconStepEnv
from xrobocon.common import load_trained_model

def train_step_loop(total_timesteps=100000, chunk_size=10000, base_model_path=None, robot_type='tristar'):
    """
    段差乗り越え訓練ループ
    
    Args:
        total_timesteps: 総ステップ数
        chunk_size: 1回の実行（Genesis再起動）あたりのステップ数
        base_model_path: ベースとなるモデルのパス
        robot_type: ロボットタイプ ('tristar', 'tristar_large', etc.)
    """
    
    # モデル保存ディレクトリとモデル名（ロボットタイプごとに分ける）
    model_dir = "."
    model_name = f"xrobocon_ppo_{robot_type}_step"
    final_model_path = os.path.join(model_dir, f"{model_name}.zip")
    
    # 現在の進捗を確認
    current_steps = 0
    start_loop = 1
    
    # 既存のステップモデルがあればそこから再開、なければベースモデル（平地）から開始
    if os.path.exists(final_model_path):
        print(f"既存の段差モデルから再開します: {final_model_path}")
        current_base_model = final_model_path
    elif base_model_path and os.path.exists(base_model_path):
        print(f"平地モデルをベースに開始します: {base_model_path}")
        current_base_model = base_model_path
    else:
        print("警告: ベースモデルが見つかりません。スクラッチから学習します。")
        current_base_model = "scratch"

    # ループ計算
    num_loops = total_timesteps // chunk_size
    
    print(f"訓練開始 ({robot_type}): 全{total_timesteps}ステップ ({chunk_size}ステップ x {num_loops}ループ)")
    
    for i in range(start_loop, num_loops + 1):
        print(f"\n{'='*60}")
        print(f"ループ {i}/{num_loops} | 累積ステップ: {(i-1)*chunk_size} -> {i*chunk_size}")
        print(f"{'='*60}")
        
        # train_rl_step.py をサブプロセスとして実行
        # メモリリーク回避のため、Genesisを毎回再起動する
        cmd = f"python train_rl_step.py --train --steps {chunk_size} --base {current_base_model} --env step --robot {robot_type} --save_name {model_name}"
        
        print(f"コマンド実行: {cmd}")
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"エラーが発生しました (Exit Code: {exit_code})。ループを中断します。")
            break
            
        # 次のループのためにベースモデルを更新
        current_base_model = final_model_path
        
        # 少し待機（メモリ解放待ち）
        print("メモリ解放待機中...")
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XROBOCON Step Climbing Training Loop')
    parser.add_argument('--steps', type=int, default=100000, help='総ステップ数')
    parser.add_argument('--chunk', type=int, default=10000, help='チャンクサイズ')
    parser.add_argument('--robot', type=str, default='tristar', help='ロボットタイプ (tristar, tristar_large)')
    # Phase 3-2b: 地形認識実装に伴い、スクラッチから学習開始
    # 入力次元が変更されたため (15 -> 40)、以前のモデルは使用不可
    base_model = None
    
    args = parser.parse_args()
    
    train_step_loop(
        total_timesteps=args.steps,
        chunk_size=args.chunk,
        base_model_path=base_model,
        robot_type=args.robot
    )
