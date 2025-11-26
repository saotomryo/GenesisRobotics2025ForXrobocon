#!/bin/bash
# Genesis ロボットシミュレーター セットアップスクリプト

echo "==================================="
echo "Genesis ロボットシミュレーター"
echo "セットアップスクリプト"
echo "==================================="

# 仮想環境の確認
if [ -d "venv" ]; then
    echo "✓ 仮想環境が既に存在します"
else
    echo "仮想環境を作成中..."
    python -m venv venv
    echo "✓ 仮想環境を作成しました"
fi

# 仮想環境を有効化
echo ""
echo "仮想環境を有効化するには、以下のコマンドを実行してください:"
echo ""
echo "  source .venv/bin/activate"
echo ""
echo "その後、依存関係をインストールしてください:"
echo ""
echo "  pip install -r requirements.txt"
echo ""
echo "シミュレーターを実行:"
echo ""
echo "  python simulator.py"
echo ""
echo "==================================="
