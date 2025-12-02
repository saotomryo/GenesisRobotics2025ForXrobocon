# XROBOCON Simulator

NHK学生ロボコン2025「XROBOCON」のシミュレーター環境。Genesisフィジックスエンジンと強化学習を使用して、ロボットの自律制御を実現します。

## 概要

このプロジェクトは、3段構造のフィールド上でコインを獲得するロボットシミュレーターです。強化学習（RL）とGemini APIを組み合わせて、戦略的な行動計画と低レベル制御を実現します。

### 主な機能

- **3段フィールド環境**: NHK学生ロボコンのルールに基づいた3段構造のフィールド
- **物理シミュレーション**: Genesisエンジンによるリアルな物理演算
- **強化学習統合**: Goal-Conditioned RLによる目標指向型制御
- **シナリオベース学習**: 複数の移動パターンを網羅的に学習
- **ゲームロジック**: コイン獲得、スコアリング、2秒滞在ルール

## 必要要件

- Python 3.10以上
- CUDA対応GPU（推奨）またはMac（MPS対応）
- 8GB以上のRAM

## インストール

1. リポジトリのクローン:
```bash
git clone <repository-url>
cd GenesisRobotics
```

2. 仮想環境の作成と有効化:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

3. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. シミュレーションの実行

基本的なシミュレーションを実行:
```bash
python run_xrobocon.py
```

このコマンドで3Dビューアが開き、フィールドとロボットが表示されます。

### 2. 強化学習の訓練

#### 基本的な訓練（Tri-starロボット）
メモリリーク対策済みの`train_loop.py`を使用することを推奨します。

```bash
# 平地移動訓練（10万ステップ）
python train_loop.py --steps 100000 --chunk 10000
```

#### 段差登坂訓練（転移学習）
平地移動モデルをベースに段差登坂を学習します。

```bash
python train_rl_step.py --steps 100000 --base_model xrobocon_ppo_tristar_flat.zip
```

### 3. 訓練済みモデルのテスト

```bash
# 訓練済みモデルの可視化
python visualize_trained_model.py
```

### 4. シナリオの確認

```bash
# 訓練シナリオの可視化
python visualize_scenarios.py
```

## プロジェクト構造

```
GenesisRobotics/
├── xrobocon/              # XROBOCONシミュレーターパッケージ
│   ├── __init__.py
│   ├── common.py          # 共通設定（MPS設定、Genesis初期化など）
│   ├── field.py           # 3段フィールドの生成
│   ├── robot.py           # ロボットクラス（2輪駆動/Tri-star）
│   ├── game.py            # ゲームロジック（スコア、コイン獲得）
│   ├── env.py             # Gym環境ラッパー（RL用）
│   └── assets/
│       ├── robot.xml      # 標準ロボット（2輪）
│       └── robot_tristar.xml # Tri-starロボット
├── run_xrobocon.py        # シミュレーション実行スクリプト
├── train_loop.py          # メモリ安全な訓練ループスクリプト
├── train_rl_step.py       # 段差登坂訓練スクリプト
├── visualize_scenarios.py # シナリオ可視化スクリプト
├── visualize_trained_model.py # モデル評価・可視化スクリプト
├── requirements.txt       # 依存パッケージ
└── README.md              # このファイル
```

## アーキテクチャ

### フィールド構造

- **Tier 1（下段）**: 半径1.85m、高さ0.6m
- **Tier 2（中段）**: 半径3.25m、高さ0.35m
- **Tier 3（上段）**: 半径4.65m、高さ0.1m
- **スロープ**: なし（段差登坂課題に変更）

### ロボット仕様

#### Tri-star Robot (New)
- **駆動方式**: Tri-starホイール機構（8モーター）
- **制御**: ハイブリッド制御
  - 平地: 独立ホイール駆動
  - 段差: フレーム回転駆動
- **自由度**: 14 DOF

#### Standard Robot (Legacy)
- **駆動方式**: 2輪差動駆動
- **自由度**: 8 DOF

### 強化学習環境

#### 観測空間（15次元）
- ロボット位置 (x, y, z): 3次元
- ロボット姿勢 (roll, pitch, yaw): 3次元
- 線形速度 (vx, vy, vz): 3次元
- 角速度 (wx, wy, wz): 3次元
- ターゲットベクトル (dx, dy, dz): 3次元

#### 行動空間（2次元/8次元）
- Standard: 左右ホイールトルク (2次元)
- Tri-star: 左右フレームトルク + 6ホイールトルク (8次元)

#### 報酬関数
- **ターゲット接近**: 距離が縮まると正の報酬
- **ターゲット到達**: 0.3m以内で大きな報酬
- **安定性**: 姿勢安定化への報酬
- **転倒/落下**: ペナルティとエピソード終了

### 訓練シナリオ

- **平地移動**: 指定距離・方向への移動
- **段差登坂**: Tier間の段差乗り越え

## 技術スタック

- **物理エンジン**: [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) - PyTorchベースの微分可能物理エンジン
- **強化学習**: [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - PPOアルゴリズム
- **環境**: [Gymnasium](https://gymnasium.farama.org/) - RL環境標準インターフェース
- **数値計算**: NumPy, PyTorch

## トラブルシューティング

### Mac (MPS) での学習時のエラー

`NotImplementedError: The operator 'aten::linalg_qr' is not currently implemented for the MPS device.`

**解決策**: 環境変数を設定してCPUフォールバックを有効化
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### ビューアが表示されない

- Mission Control (F3) で別デスクトップを確認
- Cmd+Tab でウィンドウを切り替え
- Dockでアプリケーションを確認

### 学習が遅い

- `train_rl.py`で`render_mode=None`を確認（訓練時は描画なし）
- GPUが正しく認識されているか確認: `torch.cuda.is_available()`

## 今後の開発予定

- [ ] **Phase 4: Gemini Planning**
  - カメラ画像の取得
  - Gemini APIによる戦略立案
  - 高レベルプランニングと低レベル制御の統合

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考資料

- [NHK学生ロボコン2025](https://official-robocon.com/gakusei/)
- [Genesis Documentation](https://github.com/Genesis-Embodied-AI/Genesis)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
