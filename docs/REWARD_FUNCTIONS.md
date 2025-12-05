# 報酬関数モジュール ガイド

## 概要

`xrobocon/reward_functions.py`は、強化学習の報酬関数をモジュール化したファイルです。各報酬要素を独立した関数として定義し、環境側で組み合わせて使用することで、柔軟な報酬設計を実現します。

## 設計思想

### モジュール化のメリット

1. **再利用性**: 同じ報酬関数を複数の環境で使用可能
2. **調整の容易さ**: パラメータを変更するだけで報酬バランスを調整
3. **可読性**: 各報酬要素の役割が明確
4. **テスト性**: 個別の報酬関数を独立してテスト可能

### パラメータ管理

`RewardConfig`クラスで全パラメータを一元管理し、環境側で簡単に調整できます。

## 提供される報酬関数

### 1. distance_progress_reward
**ターゲットへの接近報酬**

```python
def distance_progress_reward(current_dist, prev_dist, weight=100.0):
    return (prev_dist - current_dist) * weight
```

- **用途**: ターゲットに近づくことを奨励
- **パラメータ**:
  - `current_dist`: 現在のターゲットまでの距離
  - `prev_dist`: 前回のターゲットまでの距離
  - `weight`: 報酬の重み（デフォルト: 100.0）

### 2. proximity_bonus_reward
**ターゲット付近でのボーナス報酬**

```python
def proximity_bonus_reward(dist, threshold=1.0, max_bonus=50.0):
    if dist < threshold:
        return (threshold - dist) / threshold * max_bonus
    return 0.0
```

- **用途**: ターゲット付近に入ることを奨励
- **パラメータ**:
  - `dist`: ターゲットまでの距離
  - `threshold`: ボーナスが発生する距離閾値（デフォルト: 1.0m）
  - `max_bonus`: 最大ボーナス値（デフォルト: 50.0）

### 3. slowdown_reward
**ターゲット付近での減速報酬**

```python
def slowdown_reward(dist, speed, dist_threshold=0.7, base_speed=0.1, 
                    speed_factor=0.3, bonus_weight=30.0, penalty_weight=20.0):
    if dist >= dist_threshold:
        return 0.0
    
    target_speed = base_speed + dist * speed_factor
    
    if speed < target_speed:
        return (target_speed - speed) * bonus_weight
    else:
        return -(speed - target_speed) * penalty_weight
```

- **用途**: ターゲット付近で減速し、安定して到達することを奨励
- **パラメータ**:
  - `dist`: ターゲットまでの距離
  - `speed`: 現在の速度
  - `dist_threshold`: 減速を要求する距離閾値（デフォルト: 0.7m）
  - `base_speed`: 基準速度（最小）（デフォルト: 0.1m/s）
  - `speed_factor`: 距離に応じた速度係数（デフォルト: 0.3）
  - `bonus_weight`: ボーナス報酬の重み（デフォルト: 30.0）
  - `penalty_weight`: ペナルティの重み（デフォルト: 20.0）

**目標速度の計算**: `target_speed = base_speed + dist * speed_factor`
- 距離0.1m → 目標速度0.13m/s（ほぼ停止）
- 距離0.7m → 目標速度0.31m/s（ゆっくり）

### 4. success_reward
**成功報酬（ターゲット到達時）**

```python
def success_reward(dist, speed=None, dist_threshold=0.5, base_reward=500.0, 
                   speed_bonus_enabled=True, speed_threshold=0.3, 
                   speed_bonus_weight=100.0):
    if dist >= dist_threshold:
        return 0.0, False
    
    reward = base_reward
    
    if speed_bonus_enabled and speed is not None:
        speed_bonus = max(0, (speed_threshold - speed) * speed_bonus_weight)
        reward += speed_bonus
    
    return reward, True
```

- **用途**: ターゲット到達時の報酬（速度が低いほど高報酬）
- **パラメータ**:
  - `dist`: ターゲットまでの距離
  - `speed`: 現在の速度（Noneの場合は速度ボーナスなし）
  - `dist_threshold`: 成功判定の距離閾値（デフォルト: 0.5m）
  - `base_reward`: 基本成功報酬（デフォルト: 500.0）
  - `speed_bonus_enabled`: 速度ボーナスを有効にするか（デフォルト: True）
  - `speed_threshold`: 速度ボーナスの閾値（デフォルト: 0.3m/s）
  - `speed_bonus_weight`: 速度ボーナスの重み（デフォルト: 100.0）
- **戻り値**: `(報酬値, 成功フラグ)`

### 5. height_gain_reward
**高さ獲得報酬（段差登坂時）**

```python
def height_gain_reward(current_height, prev_height, weight=500.0):
    height_diff = current_height - prev_height
    if height_diff > 0:
        return height_diff * weight
    return 0.0
```

- **用途**: 段差を登ることを奨励
- **パラメータ**:
  - `current_height`: 現在の高さ
  - `prev_height`: 前回の高さ
  - `weight`: 報酬の重み（デフォルト: 500.0）

### 6. alignment_reward
**ターゲット方向への整列報酬**

```python
def alignment_reward(robot_yaw, target_angle, tolerance=15.0, weight=10.0):
    angle_diff = abs(((target_angle - robot_yaw + 180) % 360) - 180)
    
    if angle_diff < tolerance:
        alignment_score = (tolerance - angle_diff) / tolerance
        return alignment_score * weight
    else:
        return -(angle_diff - tolerance) * 0.1
```

- **用途**: ターゲット方向を向くことを奨励
- **パラメータ**:
  - `robot_yaw`: ロボットの向き（度）
  - `target_angle`: ターゲット方向の角度（度）
  - `tolerance`: 許容角度差（度）（デフォルト: 15.0）
  - `weight`: 報酬の重み（デフォルト: 10.0）

### 7. stability_penalty
**安定性ペナルティ（転倒防止）**

```python
def stability_penalty(roll, pitch, roll_weight=0.02, pitch_weight=0.02):
    return -(abs(roll) * roll_weight + abs(pitch) * pitch_weight)
```

- **用途**: 転倒を防ぐ
- **パラメータ**:
  - `roll`: ロール角（度）
  - `pitch`: ピッチ角（度）
  - `roll_weight`: ロールペナルティの重み（デフォルト: 0.02）
  - `pitch_weight`: ピッチペナルティの重み（デフォルト: 0.02）

### 8. speed_limit_penalty
**速度超過ペナルティ**

```python
def speed_limit_penalty(speed, max_speed=1.5, penalty_weight=2.0):
    if speed > max_speed:
        return -(speed - max_speed) * penalty_weight
    return 0.0
```

- **用途**: 過度な速度を抑制
- **パラメータ**:
  - `speed`: 現在の速度
  - `max_speed`: 最大許容速度（デフォルト: 1.5m/s）
  - `penalty_weight`: ペナルティの重み（デフォルト: 2.0）

### 9. fall_penalty
**転倒・落下ペナルティ**

```python
def fall_penalty(robot_z, roll, pitch, min_height=0.0, max_tilt=70.0, 
                 penalty=-100.0):
    if robot_z < min_height:
        return penalty, True
    
    if abs(roll) > max_tilt or abs(pitch) > max_tilt:
        return penalty, True
    
    return 0.0, False
```

- **用途**: 転倒・落下時にエピソードを終了
- **パラメータ**:
  - `robot_z`: ロボットの高さ
  - `roll`: ロール角（度）
  - `pitch`: ピッチ角（度）
  - `min_height`: 最小許容高さ（デフォルト: 0.0m）
  - `max_tilt`: 最大許容傾き（度）（デフォルト: 70.0）
  - `penalty`: ペナルティ値（デフォルト: -100.0）
- **戻り値**: `(ペナルティ値, 終了フラグ)`

## RewardConfigクラス

全パラメータを一元管理するクラス。

```python
class RewardConfig:
    def __init__(self):
        # 距離報酬
        self.distance_progress_weight = 100.0
        
        # 近接ボーナス
        self.proximity_threshold = 1.0
        self.proximity_max_bonus = 50.0
        
        # 減速報酬
        self.slowdown_dist_threshold = 0.7
        self.slowdown_base_speed = 0.1
        self.slowdown_speed_factor = 0.3
        self.slowdown_bonus_weight = 30.0
        self.slowdown_penalty_weight = 20.0
        
        # 成功報酬
        self.success_dist_threshold = 0.5
        self.success_base_reward = 500.0
        self.success_speed_bonus_enabled = True
        self.success_speed_threshold = 0.3
        self.success_speed_bonus_weight = 100.0
        
        # 高さ報酬
        self.height_gain_weight = 500.0
        
        # 整列報酬
        self.alignment_tolerance = 15.0
        self.alignment_weight = 10.0
        
        # 安定性ペナルティ
        self.stability_roll_weight = 0.02
        self.stability_pitch_weight = 0.02
        
        # 速度制限
        self.speed_limit_max = 1.5
        self.speed_limit_penalty_weight = 2.0
        
        # 転倒・落下
        self.fall_min_height = 0.0
        self.fall_max_tilt = 70.0
        self.fall_penalty = -100.0
```

## 使用例

### 基本的な使い方

```python
from xrobocon.reward_functions import (
    RewardConfig, distance_progress_reward, proximity_bonus_reward,
    slowdown_reward, success_reward, height_gain_reward
)

class MyEnv(XRoboconBaseEnv):
    def __init__(self, render_mode=None, robot_type='tristar'):
        super().__init__(render_mode, robot_type)
        
        # 報酬設定を初期化
        self.reward_config = RewardConfig()
        
        # 必要に応じてパラメータを調整
        self.reward_config.success_base_reward = 1000.0
        self.reward_config.proximity_max_bonus = 100.0
    
    def step(self, action):
        # ... アクション適用 ...
        
        reward = 0.0
        
        # 1. 距離報酬
        reward += distance_progress_reward(
            dist, 
            self.prev_dist, 
            self.reward_config.distance_progress_weight
        )
        
        # 2. 近接ボーナス
        reward += proximity_bonus_reward(
            dist,
            self.reward_config.proximity_threshold,
            self.reward_config.proximity_max_bonus
        )
        
        # 3. 減速報酬
        reward += slowdown_reward(
            dist,
            speed,
            self.reward_config.slowdown_dist_threshold,
            self.reward_config.slowdown_base_speed,
            self.reward_config.slowdown_speed_factor,
            self.reward_config.slowdown_bonus_weight,
            self.reward_config.slowdown_penalty_weight
        )
        
        # 4. 成功判定
        success_rew, terminated = success_reward(
            dist,
            speed,
            self.reward_config.success_dist_threshold,
            self.reward_config.success_base_reward,
            self.reward_config.success_speed_bonus_enabled,
            self.reward_config.success_speed_threshold,
            self.reward_config.success_speed_bonus_weight
        )
        reward += success_rew
        
        # 5. 高さ報酬（段差環境のみ）
        reward += height_gain_reward(
            robot_pos[2],
            self.prev_height,
            self.reward_config.height_gain_weight
        )
        
        return obs, reward, terminated, truncated, info
```

### パラメータ調整の例

```python
# 成功報酬を強化
self.reward_config.success_base_reward = 1000.0

# 減速をより強く奨励
self.reward_config.slowdown_bonus_weight = 50.0
self.reward_config.slowdown_penalty_weight = 30.0

# 近接ボーナスの範囲を拡大
self.reward_config.proximity_threshold = 1.5
self.reward_config.proximity_max_bonus = 100.0

# 速度ボーナスを無効化
self.reward_config.success_speed_bonus_enabled = False
```

## 報酬設計のベストプラクティス

### 1. バランスの取り方

- **成功報酬を最も高く設定**: ゴール達成が最優先
- **近接ボーナスで誘導**: ターゲット付近に入ることを奨励
- **減速報酬で安定性向上**: ターゲット付近で減速することを学習

### 2. 調整の順序

1. まず成功報酬のみで学習
2. 学習が進まない場合、近接ボーナスを追加
3. 安定性が低い場合、減速報酬を追加
4. 細かい調整（重みの調整）

### 3. デバッグ方法

各報酬要素の値をログに出力して、バランスを確認：

```python
print(f"Distance: {dist_rew:.2f}, Proximity: {prox_rew:.2f}, "
      f"Slowdown: {slow_rew:.2f}, Success: {succ_rew:.2f}")
```

## まとめ

報酬関数モジュールを使用することで：

1. **再利用性**: 同じ報酬関数を複数の環境で使用
2. **調整の容易さ**: パラメータ変更のみで報酬バランスを調整
3. **可読性**: 報酬設計が明確
4. **実験の効率化**: 異なる報酬設定を素早く試せる

学習の進捗に応じて、`RewardConfig`のパラメータを調整することで、最適な報酬設計を見つけることができます。
