# 強化学習 訓練ガイド

このガイドでは、XROBOCON Simulatorでの強化学習訓練の詳細な手順を説明します。

## 目次

1. [訓練の準備](#訓練の準備)
2. [訓練の実行](#訓練の実行)
3. [ハイパーパラメータ調整](#ハイパーパラメータ調整)
4. [訓練の監視](#訓練の監視)
5. [学習結果の評価](#学習結果の評価)
6. [トラブルシューティング](#トラブルシューティング)

## 訓練の準備

### 環境変数の設定

#### Mac (MPS) の場合

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

この設定により、MPSで未対応の演算をCPUにフォールバックします。

#### Linux/Windows (CUDA) の場合

通常は設定不要ですが、特定のGPUを指定する場合:

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU 0を使用
```

### 訓練前のチェックリスト

- [ ] 仮想環境が有効化されている
- [ ] 依存パッケージがインストール済み
- [ ] ディスク容量が十分（モデル保存用に1GB以上推奨）
- [ ] GPUメモリが十分（4GB以上推奨）

## 訓練の実行

### 基本的な訓練

```bash
python train_rl.py --train --steps 100000
```

**パラメータ:**
- `--train`: 訓練モード
- `--steps`: 総訓練ステップ数（デフォルト: 100000）

### 推奨訓練ステップ数

| 目的 | ステップ数 | 所要時間（目安） |
|------|-----------|----------------|
| 動作確認 | 1,000 | 30秒 |
| 基本動作習得 | 10,000 | 5分 |
| 実用レベル | 100,000 | 30分 |
| 高精度 | 500,000 | 2.5時間 |
| 最高性能 | 1,000,000+ | 5時間以上 |

### 訓練の中断と再開

訓練は`Ctrl+C`で中断できます。モデルは自動的に保存されます。

**再開方法:**

同じコマンドを実行すると、既存のモデルを読み込んで訓練を継続します：
```bash
python train_rl.py --train --steps 100000
```

**重要な注意点:**
- `--steps`で指定したステップ数は、**中断した地点からの追加ステップ数**です
- 例: 50,000ステップで中断した場合、上記コマンドで150,000ステップまで訓練されます
- 累積ステップ数はTensorBoardまたはコンソール出力で確認できます

**既存のモデルを破棄して最初から訓練する場合:**
```bash
# モデルファイルを削除
rm xrobocon_ppo.zip

# または別名でバックアップ
mv xrobocon_ppo.zip xrobocon_ppo_backup_$(date +%Y%m%d_%H%M%S).zip

# 新規訓練開始
python train_rl.py --train --steps 100000
```

## ハイパーパラメータ調整

### `train_rl.py`の編集

より詳細な制御が必要な場合、`train_rl.py`を編集してPPOのパラメータを調整できます:

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # 学習率
    n_steps=2048,            # バッチサイズ
    batch_size=64,           # ミニバッチサイズ
    n_epochs=10,             # エポック数
    gamma=0.99,              # 割引率
    gae_lambda=0.95,         # GAEパラメータ
    clip_range=0.2,          # クリッピング範囲
    ent_coef=0.0,            # エントロピー係数
    verbose=1,
    device="mps"
)
```

### パラメータの影響

#### `learning_rate` (学習率)
- **デフォルト**: 3e-4
- **高い値**: 学習が速いが不安定
- **低い値**: 学習が遅いが安定
- **推奨範囲**: 1e-4 ~ 1e-3

#### `n_steps` (バッチサイズ)
- **デフォルト**: 2048
- **高い値**: メモリ使用量増加、学習安定
- **低い値**: メモリ節約、学習不安定
- **推奨範囲**: 1024 ~ 4096

#### `gamma` (割引率)
- **デフォルト**: 0.99
- **高い値**: 長期的報酬を重視
- **低い値**: 短期的報酬を重視
- **推奨範囲**: 0.95 ~ 0.99

### 報酬関数のカスタマイズ

`xrobocon/env.py`の`step()`メソッドで報酬を調整:

```python
# ターゲット接近報酬の重み調整
reward += (self.prev_dist - dist) * 20.0  # デフォルト: 20.0

# ターゲット到達報酬の調整
if dist < 0.3 and z_diff < 0.2:
    reward += 50.0  # デフォルト: 50.0

# 安定性ペナルティの調整
stability_penalty = abs(euler[0]) * 0.01 + abs(euler[1]) * 0.01  # デフォルト: 0.01
```

## 訓練の監視

### TensorBoardの使用（オプション）

TensorBoardを使用して訓練を可視化できます。

1. `train_rl.py`を編集してログディレクトリを指定:

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="mps"
)
```

2. TensorBoardを起動:

```bash
tensorboard --logdir=./tensorboard_logs/
```

3. ブラウザで `http://localhost:6006` を開く

### コンソール出力の確認

訓練中、以下のような出力が表示されます:

```
-----------------------------
| time/              |      |
|    fps             | 76   |
|    iterations      | 1    |
|    time_elapsed    | 26   |
|    total_timesteps | 2048 |
-----------------------------
```

**重要な指標:**
- `fps`: シミュレーション速度（高いほど良い）
- `iterations`: 訓練イテレーション数
- `total_timesteps`: 累積ステップ数

## 訓練済みモデルのテスト

### 基本的なテスト

```bash
python train_rl.py --test
```

このコマンドで3Dビューアが開き、訓練済みエージェントの動作を確認できます。

### テスト時の観察ポイント

1. **ターゲットへの移動**: エージェントが目標に向かって移動しているか
2. **安定性**: ロボットが転倒せずに移動しているか
3. **スロープ登坂**: スロープを登れているか
4. **到達精度**: ターゲットの近くまで到達できているか

### パフォーマンス評価

複数回テストを実行して成功率を測定:

```bash
# 10回テストを実行
for i in {1..10}; do
    echo "Test $i"
    python train_rl.py --test
done
```

## 学習結果の評価

訓練が完了したら、以下の方法で学習結果を評価します。

### 1. コンソール出力の解析

訓練中に表示される統計情報から学習の進捗を確認できます。

#### 基本的な出力例

```
-----------------------------
| time/              |      |
|    fps             | 25   |
|    iterations      | 48   |
|    time_elapsed    | 1234 |
|    total_timesteps | 98304|
-----------------------------
```

#### 重要な指標の意味

| 指標 | 意味 | 良い値 | 悪い値 |
|------|------|--------|--------|
| `fps` | シミュレーション速度（フレーム/秒） | 50以上 | 20以下 |
| `iterations` | 訓練イテレーション数 | - | - |
| `total_timesteps` | 累積訓練ステップ数 | 目標値に到達 | 増加しない |
| `time_elapsed` | 経過時間（秒） | - | - |

**FPSが低い場合の原因:**
- 他のアプリケーションがGPU/CPUを使用している
- バッチサイズが大きすぎる
- 描画が有効になっている（訓練時は無効化推奨）

### 2. 訓練ログの確認

訓練中のFPS変動を確認して、パフォーマンスの問題を特定します。

```bash
# 訓練実行時の出力をファイルに保存
python train_rl.py --train --steps 100000 2>&1 | tee training_log.txt

# FPSの平均を計算
grep "Running at" training_log.txt | awk '{print $6}' | awk '{sum+=$1; count++} END {print "Average FPS:", sum/count}'
```

**期待されるFPS:**
- **Mac (MPS)**: 20-40 FPS
- **CUDA GPU (RTX 3060以上)**: 50-80 FPS
- **CPU**: 5-15 FPS

### 3. モデルファイルの確認

訓練が正常に完了すると、`xrobocon_ppo.zip`ファイルが生成されます。

```bash
# モデルファイルの存在確認
ls -lh xrobocon_ppo.zip

# 期待されるファイルサイズ: 100KB - 200KB
```

**ファイルサイズが異常に小さい場合（< 50KB）:**
- 訓練が正常に完了していない可能性
- モデルが保存されていない可能性

### 4. テスト実行による定性評価

訓練済みモデルの動作を視覚的に確認します。

```bash
python train_rl.py --test
```

#### 評価チェックリスト

**基本動作（10,000ステップ以上訓練後）:**
- [ ] ロボットがランダムに動かず、方向性のある動きをする
- [ ] 転倒せずに数秒間移動できる
- [ ] ターゲット方向に向かって移動する傾向がある

**実用レベル（100,000ステップ以上訓練後）:**
- [ ] ターゲットに向かって直進できる
- [ ] スロープを登れる（成功率50%以上）
- [ ] 転倒せずに30秒以上移動できる
- [ ] ターゲットから0.5m以内に到達できる

**高精度（500,000ステップ以上訓練後）:**
- [ ] スロープを安定して登れる（成功率80%以上）
- [ ] ターゲットから0.3m以内に到達できる
- [ ] 複数のターゲットに連続して到達できる
- [ ] 転倒率が5%以下

### 5. 定量的評価スクリプト

以下のスクリプトで成功率を自動測定できます。

#### 評価スクリプトの作成

`evaluate_model.py`を作成:

```python
import os
import numpy as np
from xrobocon.env import XRoboconEnv

def evaluate_model(model_path, num_episodes=10):
    """訓練済みモデルを評価"""
    from stable_baselines3 import PPO
    
    env = XRoboconEnv(render_mode=None)  # 描画なし（高速化）
    model = PPO.load(model_path)
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:  # 最大1000ステップ
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # ターゲット到達判定
            robot_pos = env.robot.get_pos().cpu().numpy()
            target_pos = np.array(env.current_target['pos'])
            dist = np.linalg.norm(robot_pos[:2] - target_pos[:2])
            
            if dist < 0.3:  # 0.3m以内で成功
                success_count += 1
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}, Success={'Yes' if dist < 0.3 else 'No'}")
    
    # 統計情報
    print("\n=== 評価結果 ===")
    print(f"成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"平均報酬: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均ステップ数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(episode_lengths)
    }

if __name__ == "__main__":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Mac用
    evaluate_model("xrobocon_ppo.zip", num_episodes=20)
```

#### 実行方法

```bash
python evaluate_model.py
```

#### 期待される結果

**10,000ステップ訓練後:**
```
成功率: 2/20 (10.0%)
平均報酬: -50.23 ± 30.12
平均ステップ数: 450.3 ± 200.5
```

**100,000ステップ訓練後:**
```
成功率: 12/20 (60.0%)
平均報酬: 25.45 ± 15.32
平均ステップ数: 320.1 ± 120.3
```

**500,000ステップ訓練後:**
```
成功率: 17/20 (85.0%)
平均報酬: 45.67 ± 10.21
平均ステップ数: 250.5 ± 80.2
```

### 6. 学習曲線の可視化（TensorBoard使用時）

TensorBoardを有効にしている場合、学習曲線を確認できます。

#### TensorBoardの起動

```bash
tensorboard --logdir=./tensorboard_logs/
```

ブラウザで `http://localhost:6006` を開く。

#### 確認すべきグラフ

1. **`rollout/ep_rew_mean`**: エピソード平均報酬
   - **期待される傾向**: 右肩上がり
   - **良い兆候**: 安定して増加
   - **悪い兆候**: 横ばいまたは減少

2. **`train/entropy_loss`**: エントロピー損失
   - **期待される傾向**: 徐々に減少
   - **意味**: エージェントの行動が確信的になる

3. **`train/policy_gradient_loss`**: ポリシー勾配損失
   - **期待される傾向**: 変動しながら減少
   - **意味**: ポリシーが改善されている

### 7. シナリオ別の評価

各訓練シナリオでの性能を個別に評価します。

#### シナリオ別評価スクリプト

`xrobocon/env.py`を一時的に編集して特定シナリオのみテスト:

```python
# env.pyのreset()メソッド内
scenario_type = 'start_ramp1'  # 固定シナリオ
# scenario_type = np.random.choice([...])  # この行をコメントアウト
```

各シナリオで評価:
```bash
# Scenario 1: Start -> Ramp 1
python evaluate_model.py

# Scenario 2: Ramp 1 -> Coin
# (env.pyを編集してscenario_type = 'ramp1_coin')
python evaluate_model.py

# 以下同様
```

### 8. 学習が不十分な場合の対処

評価結果が期待値に達しない場合:

#### 成功率が低い（< 30%）

**原因:**
- 訓練ステップ数が不足
- 報酬関数が適切でない
- ハイパーパラメータが不適切

**対策:**
1. 訓練ステップを増やす（2倍〜5倍）
2. 報酬の重みを調整（`xrobocon/env.py`）
3. 学習率を下げる（`learning_rate=1e-4`）

#### 報酬が増加しない

**原因:**
- 学習率が高すぎる/低すぎる
- 報酬設計に問題がある

**対策:**
1. 学習率を調整（`3e-4` → `1e-4` または `1e-3`）
2. 報酬関数を見直す（正の報酬が得られているか確認）

#### ロボットが転倒しやすい

**原因:**
- 安定性ペナルティが弱い
- 訓練データに転倒例が多い

**対策:**
1. 安定性ペナルティを強化（`0.01` → `0.05`）
2. 転倒判定を緩和（`60度` → `70度`）

### 9. 訓練結果の記録

訓練結果を記録して、後で比較できるようにします。

#### 訓練ログの保存

```bash
# 訓練実行
python train_rl.py --train --steps 100000 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# モデルのバックアップ
cp xrobocon_ppo.zip models/xrobocon_ppo_$(date +%Y%m%d_%H%M%S).zip
```

#### 評価結果の記録

```bash
# 評価実行と結果保存
python evaluate_model.py > logs/eval_$(date +%Y%m%d_%H%M%S).txt
```

#### 訓練履歴の管理

`training_history.md`を作成して記録:

```markdown
# 訓練履歴

## 2025-01-27 10:00 - 初回訓練
- ステップ数: 100,000
- 所要時間: 30分
- 成功率: 60%
- 平均報酬: 25.4
- メモ: デフォルト設定で訓練

## 2025-01-27 15:00 - 報酬調整後
- ステップ数: 100,000
- 所要時間: 30分
- 成功率: 75%
- 平均報酬: 35.2
- メモ: ターゲット接近報酬を20.0→30.0に変更
```



## トラブルシューティング

### 問題: 訓練が進まない

**症状**: `total_timesteps`が増えない

**解決策:**
1. GPUメモリを確認: `nvidia-smi` (CUDA) または アクティビティモニタ (Mac)
2. プロセスを再起動
3. バッチサイズを減らす: `n_steps=1024`

### 問題: エージェントが学習しない

**症状**: テスト時にランダムな動きをする

**解決策:**
1. 訓練ステップ数を増やす（最低10,000ステップ）
2. 学習率を調整: `learning_rate=1e-4`
3. 報酬関数を確認（正の報酬が得られているか）

### 問題: ロボットがすぐに転倒する

**症状**: エピソードが数ステップで終了

**解決策:**
1. 安定性ペナルティを強化: `stability_penalty * 0.05`
2. 転倒判定を緩和: `abs(euler[0]) > 70`（デフォルト: 60）
3. 初期位置をより安定な場所に変更

### 問題: メモリ不足エラー

**症状**: `CUDA out of memory` または `MPS out of memory`

**解決策:**
1. バッチサイズを減らす: `n_steps=1024`
2. 他のアプリケーションを閉じる
3. CPUモードで訓練: `device="cpu"`

### 問題: 訓練が遅い

**症状**: FPSが低い（< 30）

**解決策:**
1. 描画を無効化: `render_mode=None`（訓練時はデフォルトで無効）
2. 物理ステップを大きくする（精度は低下）: `dt=0.02`
3. より高性能なGPUを使用

## ベストプラクティス

### 1. 段階的な訓練

```bash
# ステップ1: 短時間テスト（1,000ステップ）
python train_rl.py --train --steps 1000

# ステップ2: 基本訓練（10,000ステップ）
python train_rl.py --train --steps 10000

# ステップ3: 本格訓練（100,000ステップ）
python train_rl.py --train --steps 100000
```

### 2. 定期的なバックアップ

```bash
# 訓練前
cp xrobocon_ppo.zip xrobocon_ppo_$(date +%Y%m%d_%H%M%S).zip

# または自動化
while true; do
    python train_rl.py --train --steps 10000
    cp xrobocon_ppo.zip backups/xrobocon_ppo_$(date +%Y%m%d_%H%M%S).zip
done
```

### 3. 複数シナリオの訓練

現在の実装では、4つのシナリオがランダムに選択されます。特定のシナリオに集中したい場合は、`xrobocon/env.py`を編集:

```python
# 特定シナリオのみ訓練
scenario_type = 'start_ramp1'  # ランダム選択を無効化
```

### 4. カリキュラム学習（高度）

簡単なタスクから難しいタスクへ段階的に訓練:

```python
# Phase 1: 平面移動のみ（coin_coin）
# Phase 2: スロープ登坂（start_ramp1）
# Phase 3: 全シナリオ
```

## 次のステップ

訓練が完了したら:

1. **Phase 4: Gemini Planning**に進む
2. カメラ画像を統合
3. 高レベル戦略立案を実装
4. 実機への転移学習（Sim-to-Real）

## 参考資料

- [Stable Baselines3 PPO Guide](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#ppo)
- [RL Hyperparameter Tuning](https://stable-baselines3.readthedocs.io/en/master/guide/tuning.html)
- [Genesis Performance Tips](https://genesis-world.readthedocs.io/en/latest/user_guide/performance.html)
