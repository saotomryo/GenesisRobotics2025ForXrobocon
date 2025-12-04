import numpy as np
from xrobocon.base_env import XRoboconBaseEnv
from xrobocon.robot_configs import get_start_height

class XRoboconStepHardEnv(XRoboconBaseEnv):
    """
    XROBOCON RL Environment (Step Climbing - Hard Mode)
    段差乗り越え特化訓練用の環境です。
    段差シナリオ80%、平地20%の割合で学習します。
    """
    
    def __init__(self, render_mode=None, robot_type='tristar'):
        super().__init__(render_mode, robot_type)
        
        # ロボット設定から開始高さを取得
        self.start_z_offset = get_start_height(robot_type, 'step')
        
        # エピソード時間を延長 (5秒 = 500ステップ)
        self.game.time_limit = 5.0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # シナリオ選択（段差特化）
        # 1. Flat Easy (平地移動の基礎) - 20%
        # 2. Ground -> Tier 3 (高さ10cm) - 40%
        # 3. Tier 3 -> Tier 2 (高さ25cm) - 40%
        
        scenario_type = np.random.choice(
            ['flat_easy', 'step_straight', 'step_tier3_to_tier2'],
            p=[0.2, 0.4, 0.4]  # 段差80%, 平地20%
        )
        
        scenarios = [
        {
            'name': 'Scenario 0: 平地移動 (Flat Easy)',
            'type': 'flat_easy',
            'start_pos': (4.97, -1.99, self.start_z_offset),
            'target_pos': (5.00, 0.00, self.start_z_offset),
            'start_yaw': 90.0,
        },
        {
            'name': 'Scenario 1: 段差直進 (Ground -> Tier 3)',
            'type': 'step_straight',
            'start_pos': (4.97, -1.99, self.start_z_offset),
            'target_pos': (5.00, 0.00, self.start_z_offset + 0.1),
            'start_yaw': 90.0,
        },
        {
            'name': 'Scenario 2: 段差間移動 (Tier 3 -> Tier 2)',
            'type': 'step_tier3_to_tier2',
            'start_pos': (5.00, 0.00, self.start_z_offset + 0.1),
            'target_pos': (5.00, 1.50, self.start_z_offset + 0.35),
            'start_yaw': 90.0,
        }
        ]
        
        # シナリオタイプに基づいて選択
        scenario = next(s for s in scenarios if s['type'] == scenario_type)
        
        start_pos = scenario['start_pos']
        target_pos = scenario['target_pos']
        start_yaw = scenario['start_yaw']
        
        # ランダム性を追加（開始位置の微調整）
        start_x = start_pos[0] + np.random.uniform(-0.1, 0.1)
        start_y = start_pos[1] + np.random.uniform(-0.1, 0.1)
        
        # シーンリセット
        self.scene.reset()
        
        # ロボットの位置設定
        self.robot.set_pose(
            pos=(start_x, start_y, start_pos[2]),
            euler_deg=(0, 0, start_yaw)
        )
        
        # ゲームリセット
        self.game.start()
        
        # ターゲット設定
        self.set_target(target_pos)
        
        # 高さトラッキング用
        self.prev_height = start_pos[2]
        
        # アクション変化率計算用
        self.last_action = None
        
        self.current_scenario_type = scenario_type
        
        return self._get_obs(), {'scenario_type': scenario_type}
    
    def step(self, action):
        # step_env.pyと同じ実装を使用
        # 共通のアクション適用
        self._apply_action(action)
        
        # 報酬計算
        reward = 0.0
        terminated = False
        truncated = False
        
        # 状態取得
        robot_pos = self.robot.get_pos().cpu().numpy()
        euler = self.robot.get_euler() # numpy array
        vel = self.robot.get_vel().cpu().numpy()
        speed = np.linalg.norm(vel)
        
        # 1. ターゲットへの接近報酬 (Goal-Conditioned)
        if self.current_target:
            target_pos = np.array(self.current_target['pos'])
            dist = np.linalg.norm(robot_pos[:2] - target_pos[:2])
            
            # 距離報酬 (平地と同じ)
            reward += (self.prev_dist - dist) * 100.0
            self.prev_dist = dist
            
            # 2. 高さ報酬 (段差を登ることを奨励)
            # 現在の高さと前回の高さの差分に報酬を与える
            z_diff = robot_pos[2] - self.prev_height
            if z_diff > 0: # 登っている時のみプラス
                # ロボット設定から高さ報酬の重みを取得
                from xrobocon.robot_configs import get_robot_config
                config = get_robot_config(self.robot_type)
                
                if config['reward_params'].get('use_specialized_rewards', False):
                    height_weight = config['reward_params']['height_gain_weight']
                else:
                    height_weight = 500.0  # デフォルト
                
                reward += z_diff * height_weight
            self.prev_height = robot_pos[2]
            
            # 2.5. 専用報酬の計算
            from xrobocon.robot_configs import get_robot_config
            config = get_robot_config(self.robot_type)
            if config['reward_params'].get('use_specialized_rewards', False):
                # ロボットタイプに応じて専用報酬を計算
                if self.robot_type in ['tristar', 'tristar_large']:
                    from xrobocon.step_env import XRoboconStepEnv
                    step_env = XRoboconStepEnv()
                    specialized_reward = step_env._calculate_tristar_climbing_rewards(robot_pos, action, config)
                elif self.robot_type == 'rocker_bogie':
                    from xrobocon.step_env import XRoboconStepEnv
                    step_env = XRoboconStepEnv()
                    specialized_reward = step_env._calculate_rocker_bogie_climbing_rewards(robot_pos, action, config)
                else:
                    specialized_reward = 0.0
                
                reward += specialized_reward
                
                # 整列報酬 (Alignment Reward)
                # ターゲット方向を向いているか
                if np.linalg.norm(robot_pos[:2] - target_pos[:2]) > 0.1:
                    target_vec = target_pos[:2] - robot_pos[:2]
                    target_angle = np.arctan2(target_vec[1], target_vec[0])
                    current_yaw = np.radians(euler[2])
                    
                    # 角度差 (-pi ~ pi)
                    angle_diff = np.arctan2(np.sin(target_angle - current_yaw), np.cos(target_angle - current_yaw))
                    
                    # 正面を向いているほど報酬 (最大 weight, 真後ろなら 0)
                    alignment = (1.0 + np.cos(angle_diff)) / 2.0
                    reward += alignment * config['reward_params'].get('alignment_reward_weight', 0.0)
            
            # 3. 成功判定
            if dist < 0.5:
                reward += 100.0
                terminated = True
        
        # 速度超過ペナルティ
        if speed > 1.5:
            reward -= (speed - 1.5) * 2.0
        
        # 5. 転倒・落下判定 (終了条件)
        if abs(euler[0]) > 70 or abs(euler[1]) > 70: # 段差なので少し許容
            reward -= 100.0
            terminated = True
            
        if robot_pos[2] < 0.0:
            reward -= 100.0
            terminated = True
            
        # 時間切れ
        if not self.game.is_running:
            truncated = True
            
        # アクション保存
        self.last_action = action.copy()
            
        return self._get_obs(), reward, terminated, truncated, {}
