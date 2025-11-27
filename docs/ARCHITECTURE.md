# XROBOCON Simulator - アーキテクチャドキュメント

## システム概要

XROBOCON Simulatorは、階層的な制御アーキテクチャを採用しています：

```
┌─────────────────────────────────────────┐
│   Gemini Planning (Phase 4 - 予定)      │
│   - 戦略立案                             │
│   - 目標シーケンス生成                   │
└──────────────┬──────────────────────────┘
               │ Target Position
               ▼
┌─────────────────────────────────────────┐
│   RL Agent (Goal-Conditioned)           │
│   - 目標位置への移動                     │
│   - 障害物回避                           │
│   - 安定性維持                           │
└──────────────┬──────────────────────────┘
               │ Wheel Torques
               ▼
┌─────────────────────────────────────────┐
│   Genesis Physics Engine                │
│   - 剛体動力学                           │
│   - 衝突検出                             │
│   - レンダリング                         │
└─────────────────────────────────────────┘
```

## コンポーネント詳細

### 1. Field (`xrobocon/field.py`)

#### クラス: `XRoboconField`

3段構造のフィールドとスロープを生成します。

**主要メソッド:**
- `build(scene)`: フィールドエンティティをシーンに追加
- `add_coin_spots(scene, spots)`: コインスポットの可視化

**フィールド構造:**
```python
self.tiers = [
    {'radius': 1.5, 'height': 0.1, 'z': 0.05},   # Tier 1
    {'radius': 1.0, 'height': 0.25, 'z': 0.225}, # Tier 2
    {'radius': 0.5, 'height': 0.25, 'z': 0.475}, # Tier 3
]
```

**スロープ配置:**
- Ramp 1: Ground (0.0m) → Tier 1 (0.1m), Angle 0°
- Ramp 2: Tier 1 (0.1m) → Tier 2 (0.35m), Angle 120°
- Ramp 3: Tier 2 (0.35m) → Tier 3 (0.6m), Angle 240°

### 2. Robot (`xrobocon/robot.py`)

#### クラス: `XRoboconRobot`

2輪差動駆動ロボットの制御クラス。

**主要メソッド:**
- `set_wheel_torques(left, right)`: ホイールトルク設定
- `set_pose(pos, euler_deg)`: 位置・姿勢設定
- `get_pos()`: 位置取得
- `get_euler()`: オイラー角取得（度）
- `get_vel()`: 線形速度取得
- `get_ang_vel()`: 角速度取得

**ロボットモデル (`assets/robot.xml`):**
- ベース: 円筒形（半径0.2m、高さ0.05m、質量5kg）
- ホイール: 2輪（半径0.05m、質量1kg、摩擦係数1.0）
- キャスター: 前後2個（安定性向上）
- モーター: ギア比50、制御範囲[-1.0, 1.0]

### 3. Game Logic (`xrobocon/game.py`)

#### クラス: `XRoboconGame`

ゲームルールとスコアリングを管理。

**主要メソッド:**
- `start()`: ゲーム開始
- `update(dt)`: 状態更新（毎フレーム）
- `get_info()`: ゲーム情報取得

**コインスポット配置:**
```python
# Tier 1: 8スポット（45度間隔、半径1.25m）
# Tier 2: 8スポット（45度間隔、半径0.75m）
# Tier 3: 4スポット（90度間隔、半径0.35m）
```

**スコアリング:**
- Tier 1: 1点/コイン
- Tier 2: 2点/コイン
- Tier 3: 5点/コイン（2秒滞在必要）

### 4. RL Environment (`xrobocon/env.py`)

#### クラス: `XRoboconEnv(gym.Env)`

Gymnasium互換のRL環境。

**観測空間:**
```python
observation_space = spaces.Box(
    low=-np.inf, 
    high=np.inf, 
    shape=(15,),  # [pos(3), euler(3), vel(3), ang_vel(3), target_vec(3)]
    dtype=np.float32
)
```

**行動空間:**
```python
action_space = spaces.Box(
    low=-1.0, 
    high=1.0, 
    shape=(2,),  # [left_torque, right_torque]
    dtype=np.float32
)
```

**訓練シナリオ:**

| シナリオ | 開始位置 | 目標位置 | 目的 |
|---------|---------|---------|------|
| `start_ramp1` | (1.5, -2.5, 0.1) | (1.5, 0.0, 0.1) | スロープ登坂 |
| `ramp1_coin` | (1.5, 0.0, 0.1) | Tier 1 Coin | コイン獲得 |
| `coin_coin` | Tier 1 Coin | Tier 1 Coin | 平面移動 |
| `coin_ramp2` | Tier 1 Coin | Ramp 2 Entry | 次段への移動 |

**報酬設計:**
```python
reward = 0.0

# 1. ターゲット接近報酬
reward += (prev_dist - current_dist) * 20.0

# 2. ターゲット到達報酬
if dist < 0.3 and z_diff < 0.2:
    reward += 50.0

# 3. 安定性ペナルティ
reward -= (abs(roll) + abs(pitch)) * 0.01

# 4. 転倒ペナルティ
if abs(roll) > 60 or abs(pitch) > 60:
    reward -= 100.0
    terminated = True
```

### 5. Training Script (`train_rl.py`)

Stable Baselines3のPPOアルゴリズムを使用。

**ハイパーパラメータ:**
```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="mps"  # または "cuda", "cpu"
)
```

**使用方法:**
```bash
# 訓練
python train_rl.py --train --steps 100000

# テスト
python train_rl.py --test
```

## データフロー

### 訓練時

```
1. env.reset()
   ↓
2. シナリオ選択（ランダム）
   ↓
3. ロボット位置・ターゲット設定
   ↓
4. 観測取得 → Agent
   ↓
5. Agent → 行動（トルク）
   ↓
6. Physics Step
   ↓
7. 報酬計算
   ↓
8. 終了判定
   ↓
9. ステップ3に戻る（または新エピソード）
```

### 推論時（予定）

```
1. カメラ画像取得
   ↓
2. Gemini API → 戦略立案
   ↓
3. 目標シーケンス生成
   ↓
4. 各目標に対して:
   - env.set_target(pos)
   - RL Agent実行
   - 到達確認
   ↓
5. 次の目標へ
```

## 座標系

### ワールド座標系
- X軸: 右方向（正）
- Y軸: 前方向（正）
- Z軸: 上方向（正）
- 原点: フィールド中心、地面

### ロボット座標系
- オイラー角: (Roll, Pitch, Yaw) [度]
- Yaw = 0°: X軸正方向
- Yaw = 90°: Y軸正方向

### スロープ角度
- Ramp 1: Yaw = 0° (X軸正方向)
- Ramp 2: Yaw = 120°
- Ramp 3: Yaw = 240°

## パフォーマンス最適化

### 訓練時
- `render_mode=None`: 描画無効化
- `dt=0.01`: 物理ステップ10ms
- バッチサイズ: 2048（PPOデフォルト）

### 推論時
- `render_mode="human"`: 可視化有効
- FPS: 60-85（GPU依存）

## 拡張性

### 新しいシナリオの追加

`xrobocon/env.py`の`reset()`メソッドに追加:

```python
elif scenario_type == 'new_scenario':
    start_pos = (x, y, z)
    start_euler = (roll, pitch, yaw)
    target_pos = (tx, ty, tz)
```

### 報酬関数のカスタマイズ

`xrobocon/env.py`の`step()`メソッドを編集:

```python
# カスタム報酬を追加
reward += custom_reward_function(state)
```

### 新しいロボットモデル

`xrobocon/assets/`に新しいMJCFファイルを配置し、`robot.py`で読み込み:

```python
robot_path = os.path.join(assets_dir, 'new_robot.xml')
```

## デバッグ

### ログ出力

```python
# ゲーム情報
info = game.get_info()
print(f"Score: {info['score']}, Time: {info['time']}")

# ロボット状態
pos = robot.get_pos()
euler = robot.get_euler()
print(f"Pos: {pos}, Euler: {euler}")
```

### ビジュアルデバッグ

```python
# シミュレーション実行（可視化あり）
python run_xrobocon.py

# テストモード（訓練済みモデル）
python train_rl.py --test
```

## 既知の制限事項

1. **カメラ画像**: 現在未実装（Phase 4で対応予定）
2. **マルチエージェント**: 単一ロボットのみ対応
3. **動的障害物**: 静的フィールドのみ
4. **リアルタイム性**: シミュレーション速度は実時間より速い

## 参考文献

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Gymnasium API](https://gymnasium.farama.org/api/env/)
