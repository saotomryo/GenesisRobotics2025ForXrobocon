"""
Genesis ロボットシミュレーターの設定パラメータ
"""
import torch

# カメラ設定
CAMERA_CONFIG = {
    'pos': (3.5, 0.0, 2.5),
    'lookat': (0.0, 0.0, 0.5),
    'fov': 30,
    'resolution': (800, 600),  # 解像度を下げてパフォーマンス向上
    'max_fps': 30,  # FPS上限を下げて安定性向上
}

# 物理設定
PHYSICS_CONFIG = {
    'dt': 0.01,  # タイムステップ（秒）
    'gravity': (0.0, 0.0, -9.8),  # 重力ベクトル (m/s^2)
}

# ロボット設定
ROBOT_CONFIG = {
    'model_path': 'xml/franka_emika_panda/panda.xml',  # Genesisモデルへの相対パス
    'initial_position': (0.0, 0.0, 0.0),
    'initial_euler': (0, 0, 0),  # 回転（度）
}

# Franka Panda ホームポジション (7自由度 + グリッパー2)
# 関節順序: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3, wrist_roll, finger1, finger2]
HOME_POSE = torch.tensor([
    0.0,    # 関節1: ベース回転
    -0.5,   # 関節2: 肩
    0.0,    # 関節3: 肘
    -2.0,   # 関節4: 手首1
    0.0,    # 関節5: 手首2
    1.5,    # 関節6: 手首3
    0.78,   # 関節7: 手首ロール
])

# 制御設定
CONTROL_CONFIG = {
    'position_step': 0.1,  # キー押下ごとのラジアン
    'velocity_limit': 2.0,  # 最大関節速度 (rad/s)
    'gripper_open': 0.04,   # グリッパー開位置
    'gripper_closed': 0.0,  # グリッパー閉位置
}

# シーンオブジェクト
SCENE_OBJECTS = [
    {
        'type': 'box',
        'name': 'red_box',
        'pos': (0.5, 0.0, 0.5),
        'size': (0.05, 0.05, 0.05),
        'color': (1.0, 0.0, 0.0),
    },
    {
        'type': 'box',
        'name': 'blue_box',
        'pos': (0.4, 0.2, 0.5),
        'size': (0.06, 0.06, 0.06),
        'color': (0.0, 0.0, 1.0),
    },
    {
        'type': 'sphere',
        'name': 'green_sphere',
        'pos': (0.3, -0.2, 0.5),
        'radius': 0.03,
        'color': (0.0, 1.0, 0.0),
    },
]

# 可視化設定
VISUAL_CONFIG = {
    'show_frames': False,  # 座標フレームを表示
    'show_contacts': False,  # 接触点を表示
    'ambient_light': 0.5,
}
