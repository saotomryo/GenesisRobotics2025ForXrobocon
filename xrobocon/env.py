import gymnasium as gym
from gymnasium import spaces
import numpy as np
import genesis as gs
from xrobocon.field import XRoboconField
from xrobocon.robot import XRoboconRobot
from xrobocon.game import XRoboconGame

class XRoboconEnv(gym.Env):
    """XROBOCON RL Environment"""
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.visualize = render_mode == "human"
        
        # Genesis初期化
        gs.init(backend=gs.gpu)
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.8),
            ),
            show_viewer=self.visualize,
        )
        
        # 地面
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        # フィールド
        self.field = XRoboconField()
        self.field.build(self.scene)
        
        # ロボット (初期位置はresetで設定) - フィールドから離れた訓練エリア
        self.robot = XRoboconRobot(self.scene, pos=(5.0, -1.0, 0.0), euler=(0, 0, 90))
        
        # ゲームロジック
        self.game = XRoboconGame(self.field, self.robot)
        self.field.add_coin_spots(self.scene, self.game.spots)
        
        self.scene.build()
        self.robot.post_build()
        
        # Action Space: 左右ホイールのトルク (-1.0 ~ 1.0) -> 最大トルクにスケーリング
        self.max_torque = 20.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space:
        # - Robot Pos (3)
        # - Robot Euler (3)
        # - Robot Lin Vel (3)
        # - Robot Ang Vel (3)
        # - Target Vector (3) - 最も近い未獲得コインへのベクトル
        # Total: 15
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.current_target = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # シナリオ選択
        # Phase 2: スロープ登坂訓練
        # 段階的な難易度:
        # 1. スロープ途中から（簡単）
        # 2. スロープ下から（中程度）
        # 3. 地面からTier 1まで（難しい）
        
        scenario_type = np.random.choice([
            'slope_mid',        # スロープ途中から
            'slope_bottom',     # スロープ下から
            'slope_full',       # 地面からTier 1まで
        ])
        
        if scenario_type == 'slope_mid':
            # スロープの途中から開始（最も簡単）
            # Ramp 1の中間地点
            start_pos = (1.3, -0.3, 0.03)
            start_euler = (0, 0, 90)  # Y軸正方向を向く
            target_pos = (1.5, 0.0, 0.1)  # Tier 1の端
            
        elif scenario_type == 'slope_bottom':
            # スロープの下から（中程度）
            start_pos = (1.2, -0.5, 0.0)
            start_euler = (0, 0, 90)
            target_pos = (1.5, 0.0, 0.1)
            
        elif scenario_type == 'slope_full':
            # 地面からTier 1まで（最も難しい）
            start_pos = (1.0, -1.0, 0.0)
            start_euler = (0, 0, 90)
            target_pos = (1.5, 0.5, 0.1)  # Tier 1の中央
            
        # ランダム性を少し加える（±5cm、±5度）
        # スロープは難しいので、ランダム性を減らす
        start_x = start_pos[0] + np.random.uniform(-0.05, 0.05)
        start_y = start_pos[1] + np.random.uniform(-0.05, 0.05)
        start_yaw = start_euler[2] + np.random.uniform(-5, 5)
        
        self.robot.set_pose(
            pos=(start_x, start_y, start_pos[2]),
            euler_deg=(0, 0, start_yaw)
        )
        
        # ゲームリセット
        self.game.start()
        
        # ターゲット設定
        self.set_target(target_pos)
        
        # 高さトラッキング用（報酬計算に使用）
        self.prev_height = start_pos[2]
        
        self.scene.reset()
        
        return self._get_obs(), {}
        
    def _set_random_target(self):
        """(廃止予定) ランダムなターゲットを設定"""
        pass

    def set_target(self, target_pos):
        """外部からターゲットを指定"""
        self.current_target = {'pos': target_pos, 'tier': 0}
        
        robot_pos = self.robot.get_pos().cpu().numpy()
        self.prev_dist = np.linalg.norm(robot_pos[:2] - np.array(target_pos)[:2])
        
    def step(self, action):
        # Action適用
        left_torque = action[0] * self.max_torque
        right_torque = action[1] * self.max_torque
        self.robot.set_wheel_torques(left_torque, right_torque)
        
        # 物理ステップ
        self.scene.step()
        
        # ゲーム更新
        self.game.update(0.01)
        
        # 報酬計算
        reward = 0.0
        terminated = False
        truncated = False
        
        # 状態取得
        robot_pos = self.robot.get_pos().cpu().numpy()
        euler = self.robot.get_euler() # numpy array
        
        # 1. ターゲットへの接近報酬 (Goal-Conditioned)
        if self.current_target:
            target_pos = np.array(self.current_target['pos'])
            dist = np.linalg.norm(robot_pos[:2] - target_pos[:2])
            
            # 距離が縮まったらプラス、離れたらマイナス (Shaping)
            # 重みを20.0→50.0に増加して、接近の重要性を強調
            reward += (self.prev_dist - dist) * 50.0
            self.prev_dist = dist
            
            # 高さ報酬を追加（スロープ登坂を促進）
            height_gain = robot_pos[2] - self.prev_height
            reward += height_gain * 100.0  # 高さ1m登ると+100報酬
            self.prev_height = robot_pos[2]
            
            # ターゲット到達判定 (距離0.35m以内かつ高さが近い)
            # ロボット半径0.1mを考慮して、0.35mに設定
            z_diff = abs(robot_pos[2] - target_pos[2])
            if dist < 0.35 and z_diff < 0.2:
                reward += 200.0 # 到達ボーナス（50.0→200.0に増加）
                terminated = True  # 到達したらエピソード終了（成功として記録）
                # 学習中は到達したら次のターゲットへ (Curriculum)
                # またはエピソード終了にするか？ -> 継続して別のターゲットへ向かわせる方が効率的
                # self.game._collect_spot(self.current_target) # 強制獲得扱い
                # self._set_random_target() # 次のターゲット
            
        # 2. 安定性報酬 (転倒防止) - スロープ用に強化
        # Roll/Pitchが0に近いほど良い
        # スロープでは傾きやすいので、ペナルティを5倍に強化
        stability_penalty = abs(euler[0]) * 0.05 + abs(euler[1]) * 0.05
        reward -= stability_penalty
        
        # 3. 転倒・落下判定 (終了条件)
        if abs(euler[0]) > 60 or abs(euler[1]) > 60:
            reward -= 100.0
            terminated = True
            
        if robot_pos[2] < 0.0:
            reward -= 100.0
            terminated = True
            
        # 時間切れ
        if not self.game.is_running:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
        
    def _get_obs(self):
        pos = self.robot.get_pos().cpu().numpy()
        euler = self.robot.get_euler() # numpy array
        vel = self.robot.get_vel().cpu().numpy()
        ang_vel = self.robot.get_ang_vel().cpu().numpy()
        
        # ターゲットベクトル (相対位置)
        target_vec = np.zeros(3)
        if self.current_target:
            target_pos = np.array(self.current_target['pos'])
            target_vec = target_pos - pos
            
        obs = np.concatenate([pos, euler, vel, ang_vel, target_vec])
        return obs.astype(np.float32)
        
    def _update_target(self):
        # 古いメソッドは廃止 (Goal-Conditionedでは勝手にターゲットを変えない)
        pass
