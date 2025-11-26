"""
デバッグ用シミュレーター - ビューア表示確認
"""
import genesis as gs
import torch
import time

print("="*60)
print("Genesis ビューア表示テスト")
print("="*60)

# 初期化
print("\n1. Genesis初期化中...")
gs.init(backend=gs.gpu, precision='32')
print("   ✓ 初期化完了")

# シーン作成
print("\n2. シーン作成中...")
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -1.5, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        res=(640, 480),
        max_FPS=30,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        gravity=(0.0, 0.0, -9.8),
    ),
    show_viewer=True,  # 明示的にビューアを表示
)
print("   ✓ シーン作成完了")

# 地面のみ追加
print("\n3. エンティティ追加中...")
plane = scene.add_entity(gs.morphs.Plane())
print("   ✓ 地面追加完了")

# ロボット追加
print("\n4. ロボット読み込み中...")
robot = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=(0.0, 0.0, 0.0),
        euler=(0, 0, 0),
    )
)
print("   ✓ ロボット追加完了")

# ビルド
print("\n5. シーンビルド中...")
scene.build()
print("   ✓ ビルド完了")

# 初期姿勢設定
HOME_POSE = torch.tensor([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.78])
current_pose = HOME_POSE.clone().to(gs.device)
n_dofs = robot.n_dofs

if len(current_pose) < n_dofs:
    padding = torch.zeros(n_dofs - len(current_pose), device=gs.device)
    current_pose = torch.cat([current_pose, padding])

robot.set_dofs_position(current_pose)

print("\n" + "="*60)
print("ビューアウィンドウが開いているはずです！")
print("ウィンドウが見つからない場合:")
print("  1. Dockでアプリケーションを確認")
print("  2. Mission Control (F3) で別デスクトップを確認")
print("  3. Cmd+Tab でウィンドウ切り替え")
print("="*60)
print("\nシミュレーション開始（10秒間）...")
print("ロボットがゆっくり動きます\n")

# 10秒間のテスト実行
import numpy as np
start_time = time.time()
step = 0

try:
    while time.time() - start_time < 10:
        # ゆっくり動かす
        if step % 50 == 0:
            angle = np.sin(step * 0.01) * 0.3
            current_pose[0] = angle
            robot.set_dofs_position(current_pose)
        
        scene.step()
        step += 1
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"経過時間: {elapsed:.1f}秒 / ステップ: {step}")
    
    print("\n10秒経過しました。")
    print("ウィンドウが見えましたか？ (Y/N)")
    print("\nプログラムを終了するには Ctrl+C を押してください...")
    
    # ウィンドウを開いたまま待機
    while True:
        scene.step()
        time.sleep(0.03)
        
except KeyboardInterrupt:
    print("\n\n終了しました")

print("="*60)
