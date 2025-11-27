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

#### 短時間テスト（1000ステップ）
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Mac使用時
python train_rl.py --train --steps 1000
```

#### 本格的な訓練（10万ステップ）
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Mac使用時
python train_rl.py --train --steps 100000
```

### 3. 訓練済みモデルのテスト

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Mac使用時
python train_rl.py --test
```

## プロジェクト構造

```
GenesisRobotics/
├── xrobocon/              # XROBOCONシミュレーターパッケージ
│   ├── __init__.py
│   ├── field.py           # 3段フィールドの生成
│   ├── robot.py           # ロボットクラス（2輪駆動）
│   ├── game.py            # ゲームロジック（スコア、コイン獲得）
│   ├── env.py             # Gym環境ラッパー（RL用）
│   └── assets/
│       └── robot.xml      # ロボットのMJCFモデル
├── run_xrobocon.py        # シミュレーション実行スクリプト
├── train_rl.py            # RL訓練スクリプト
├── requirements.txt       # 依存パッケージ
└── README.md              # このファイル
```

## アーキテクチャ

### フィールド構造

- **Tier 1（下段）**: 半径1.5m、高さ0.1m、8コインスポット（各1点）
- **Tier 2（中段）**: 半径1.0m、高さ0.25m、8コインスポット（各2点）
- **Tier 3（上段）**: 半径0.5m、高さ0.25m、4コインスポット（各5点、2秒滞在必要）
- **スロープ**: 各段を接続する3本のスロープ

### ロボット仕様

- **駆動方式**: 2輪差動駆動
- **センサー**: 位置、速度、姿勢（オイラー角）
- **制御**: トルク制御（左右独立）
- **自由度**: 8 DOF（フリージョイント6 + ホイール2）

### 強化学習環境

#### 観測空間（15次元）
- ロボット位置 (x, y, z): 3次元
- ロボット姿勢 (roll, pitch, yaw): 3次元
- 線形速度 (vx, vy, vz): 3次元
- 角速度 (wx, wy, wz): 3次元
- ターゲットベクトル (dx, dy, dz): 3次元

#### 行動空間（2次元）
- 左ホイールトルク: [-1.0, 1.0]
- 右ホイールトルク: [-1.0, 1.0]

#### 報酬関数
- **ターゲット接近**: 距離が縮まると正の報酬（×20.0）
- **ターゲット到達**: 0.3m以内で+50.0
- **安定性**: Roll/Pitchが小さいほど良い（ペナルティ×0.01）
- **転倒**: Roll/Pitch > 60度で-100.0（終了）
- **落下**: Z < 0で-100.0（終了）

### 訓練シナリオ

エピソードごとに以下の4パターンからランダムに選択:

1. **Start → Ramp 1**: スタート地点から最初のスロープへ
2. **Ramp 1 → Coin**: スロープ頂上から近くのコインへ
3. **Coin → Coin**: コイン間の移動（Tier 1内）
4. **Coin → Ramp 2**: コインから次のスロープへ

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
