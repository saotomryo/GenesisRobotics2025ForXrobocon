# Genesis ロボットシミュレーター

Genesisフィジックスエンジンを使用したロボットシミュレーター。Franka Emika Pandaロボットアーム、インタラクティブ制御、物理シミュレーション機能を搭載。

## 機能

- **Franka Emika Pandaロボット**: 7自由度のロボットアームとグリッパー
- **インタラクティブ制御**: キーボードベースの関節制御
- **物理シミュレーション**: 重力を含むリアルタイム剛体動力学
- **3D可視化**: 調整可能なカメラビューとレンダリング
- **サンプルオブジェクト**: 操作テスト用のボックスと球体

## インストール

1. 仮想環境を作成して有効化:
```bash
python -m venv .venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows
```

2. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

3. 最適なパフォーマンスにはCUDA対応GPUが必要です（CPUバックエンドも利用可能）

## 使用方法

### 基本シミュレーター

メインシミュレーターを実行（3Dビジュアライザーが開きます）:
```bash
python simulator.py
```

**注意**: ビジュアライザーウィンドウが開き、ロボットの動作が表示されます。ウィンドウを閉じるか、ターミナルで`Ctrl+C`を押すと終了します。

### キーボード操作

- **矢印キー**: 異なるロボット関節を制御
  - `↑/↓`: 関節1（ベース回転）
  - `←/→`: 関節2（肩）
- **数字キー 1-7**: 制御する関節を選択
- **+/-**: 選択した関節の角度を増減
- **R**: ホームポジションにリセット
- **G**: グリッパーの開閉を切り替え
- **ESC**: シミュレーター終了

### サンプルスクリプト

サンプルスクリプトを実行:
```bash
# 基本制御のデモンストレーション
python examples/basic_control.py

# オブジェクトとの相互作用と把持
python examples/object_interaction.py
```

## 設定

`config.py`を編集してカスタマイズ:
- カメラ位置と視野角
- 物理パラメータ（タイムステップ、重力）
- ロボットの初期姿勢
- 制御リミットと速度

## プロジェクト構造

```
GenesisRobotics/
├── simulator.py          # メインシミュレーターアプリケーション
├── config.py            # 設定パラメータ
├── requirements.txt     # Python依存関係
├── README.md           # このファイル
├── venv/               # 仮想環境（作成後）
└── examples/           # サンプルスクリプト
    ├── basic_control.py
    └── object_interaction.py
```

## トラブルシューティング

**ロボットモデルが見つからない**: GenesisにはFranka Pandaモデルが含まれているはずです。問題が発生した場合は、Genesisのインストールとモデルパスを確認してください。

**GPUエラー**: CUDA GPUがない場合は、`simulator.py`を修正してCPUバックエンドを使用してください:
```python
gs.init(backend=gs.cpu, precision='32')
```

**パフォーマンスの問題**: `config.py`でビューアの解像度を下げるか、目標FPSを下げてください。

リソース制約がある場合は、軽量版を使用:
```bash
python simulator_lite.py  # オブジェクトなし、低解像度
```

解像度の調整（`config.py`）:
```python
CAMERA_CONFIG = {
    'resolution': (640, 480),  # より低解像度に
    'max_fps': 30,  # FPS上限を下げる
}
```

## 参考資料

- [Genesis ドキュメント](https://github.com/Genesis-Embodied-AI/Genesis)
- [Franka Emika Panda](https://www.franka.de/)
