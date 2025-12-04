# Rocker-Bogie型ロボットの段差乗り越え報酬設計提案

## 現状の課題

現在の`rocker_bogie`設定:
- `use_specialized_rewards: False` - 基本報酬のみ
- アクション空間: 2次元 [left_drive, right_drive]
- 物理的には段差を登れる設計だが、学習が難しい可能性

## Rocker-Bogieの特徴

1. **6輪サスペンション**: 前輪・中輪・後輪が独立して動く
2. **受動的サスペンション**: ロッカーアームとボギーアームが自動的に地形に追従
3. **高い段差乗り越え能力**: 車輪直径の1.5倍程度まで登坂可能
4. **安定性**: 重心が低く、転倒しにくい

## 提案する報酬設計

### 1. **正面アプローチ報酬** (最重要)
段差に対して正面から入ることを奨励

```python
'alignment_reward_weight': 3.0,  # ターゲット方向への整列報酬
'alignment_tolerance': 15.0,     # 許容角度誤差（度）
```

**理由**: 
- Rocker-Bogieは正面から段差に入ると、前輪→中輪→後輪の順に登るため安定
- 斜めから入ると片側だけが登り、転倒リスクが高い

### 2. **速度制御報酬**
段差接近時の適切な速度を奨励

```python
'approach_speed_weight': 2.0,     # 接近速度報酬
'optimal_approach_speed': 0.3,    # 最適接近速度 (m/s)
'distance_threshold': 0.5,        # 速度制御開始距離 (m)
```

**理由**:
- 速すぎると前輪が段差に激突して跳ね返る
- 遅すぎるとトルク不足で登れない
- 適度な速度（0.3m/s程度）が最適

### 3. **前輪接地報酬**
前輪が段差に接触している状態を奨励

```python
'front_wheel_contact_weight': 1.5,  # 前輪接地報酬
'wheel_contact_bonus': 0.1,         # 接地時のボーナス
```

**理由**:
- 前輪が段差に接触していることが登坂の第一歩
- 接触を維持しながら推進力を与えることが重要

### 4. **ピッチ角報酬**
段差登坂時の適切な前傾姿勢を奨励

```python
'pitch_reward_weight': 2.0,      # ピッチ角報酬
'target_pitch_near': 15.0,       # 段差接近時の目標ピッチ角（度）
'target_pitch_far': 0.0,         # 平地での目標ピッチ角（度）
```

**理由**:
- 段差登坂時は自然に前傾する
- 適度な前傾（15度程度）を維持することで安定登坂

### 5. **高さ獲得報酬の強化**

```python
'height_gain_weight': 1000.0,    # 高さ獲得報酬（デフォルト500から増加）
'height_gain_bonus': 50.0,       # 一定高さ到達時のボーナス
```

**理由**:
- Rocker-Bogieは段差を登る能力があるため、高さ獲得を強く奨励
- 段差の各段階（Tier 1, 2, 3）到達時にボーナス

### 6. **安定性ペナルティ**

```python
'roll_penalty_weight': 3.0,       # ロール角ペナルティ
'max_safe_roll': 20.0,            # 安全なロール角（度）
'z_velocity_penalty_weight': 2.0, # Z軸速度ペナルティ（ジャンプ抑制）
```

**理由**:
- 横転を防ぐ
- ジャンプ動作を抑制（物理的に登ることを奨励）

### 7. **継続的推進力報酬**

```python
'forward_progress_weight': 1.0,   # 前進報酬
'action_smoothness_weight': 0.5,  # アクション平滑化報酬
```

**理由**:
- 段差登坂中も継続的に推進力を与えることが重要
- 急激なアクション変化を抑制

## 実装例

```python
'rocker_bogie': {
    # ... 既存の設定 ...
    
    'reward_params': {
        'use_specialized_rewards': True,  # 専用報酬を有効化
        
        # 1. 正面アプローチ報酬
        'alignment_reward_weight': 3.0,
        'alignment_tolerance': 15.0,
        
        # 2. 速度制御報酬
        'approach_speed_weight': 2.0,
        'optimal_approach_speed': 0.3,
        'distance_threshold': 0.5,
        
        # 3. 前輪接地報酬
        'front_wheel_contact_weight': 1.5,
        'wheel_contact_bonus': 0.1,
        
        # 4. ピッチ角報酬
        'pitch_reward_weight': 2.0,
        'target_pitch_near': 15.0,
        'target_pitch_far': 0.0,
        
        # 5. 高さ獲得報酬
        'height_gain_weight': 1000.0,
        'height_gain_bonus': 50.0,
        
        # 6. 安定性ペナルティ
        'roll_penalty_weight': 3.0,
        'max_safe_roll': 20.0,
        'z_velocity_penalty_weight': 2.0,
        
        # 7. 継続的推進力報酬
        'forward_progress_weight': 1.0,
        'action_smoothness_weight': 0.5,
    }
}
```

## 優先順位

学習の初期段階では以下の順で報酬を追加することを推奨:

1. **Phase 1**: 正面アプローチ + 高さ獲得
2. **Phase 2**: 速度制御 + ピッチ角
3. **Phase 3**: 前輪接地 + 安定性ペナルティ
4. **Phase 4**: 継続的推進力 + 平滑化

## 期待される効果

- **正面アプローチ**: 段差に対して垂直に進入する行動を学習
- **適切な速度**: 段差接近時に減速し、適度な速度で登坂
- **安定登坂**: 前輪→中輪→後輪の順に段差を登る自然な動作
- **高成功率**: 物理的特性を活かした効率的な段差乗り越え

## 注意点

- Rocker-Bogieは受動的サスペンションのため、アクティブな姿勢制御は不要
- 左右の駆動力のバランスが重要（正面アプローチのため）
- 段差の高さに応じて報酬の重みを調整する必要がある
