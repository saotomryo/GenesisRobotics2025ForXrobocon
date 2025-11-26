import numpy as np
import time

class XRoboconGame:
    """XROBOCON ゲームロジック管理クラス"""
    
    def __init__(self, field, robot):
        self.field = field
        self.robot = robot
        
        # ゲーム設定
        self.time_limit = 180.0 # 3分
        self.coin_radius = 0.2  # コインスポットの判定半径
        self.upper_tier_stay_time = 2.0 # 上段での滞在必要時間
        
        # 状態変数
        self.score = 0
        self.elapsed_time = 0.0
        self.start_time = None
        self.is_running = False
        
        # コインスポットの状態
        # {'id': 0, 'pos': (x, y, z), 'tier': 1, 'collected': False, 'stay_timer': 0.0}
        self.spots = []
        self._init_spots()
        
    def _init_spots(self):
        """コインスポットの初期化"""
        self.spots = []
        spot_id = 0
        
        # 下段 (Tier 1): 8個
        # R=1.25 (1.0 ~ 1.5の中間)
        for i in range(8):
            angle = np.radians(i * (360/8))
            self.spots.append({
                'id': spot_id,
                'pos': (1.25 * np.cos(angle), 1.25 * np.sin(angle), 0.105), # Zは床+微小高さ
                'tier': 1,
                'collected': False,
                'stay_timer': 0.0
            })
            spot_id += 1
            
        # 中段 (Tier 2): 8個
        # R=0.75 (0.5 ~ 1.0の中間)
        for i in range(8):
            angle = np.radians(i * (360/8) + 22.5) # 位相をずらす
            self.spots.append({
                'id': spot_id,
                'pos': (0.75 * np.cos(angle), 0.75 * np.sin(angle), 0.355),
                'tier': 2,
                'collected': False,
                'stay_timer': 0.0
            })
            spot_id += 1
            
        # 上段 (Tier 3): 4個
        # R=0.25 (0 ~ 0.5の中間)
        for i in range(4):
            angle = np.radians(i * (360/4))
            self.spots.append({
                'id': spot_id,
                'pos': (0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.605),
                'tier': 3,
                'collected': False,
                'stay_timer': 0.0
            })
            spot_id += 1
            
    def start(self):
        """ゲーム開始"""
        self.start_time = time.time()
        self.is_running = True
        self.score = 0
        self.elapsed_time = 0.0
        self._init_spots()
        print("Game Started!")
        
    def update(self, dt):
        """ゲーム状態の更新 (毎フレーム呼び出す)"""
        if not self.is_running:
            return
            
        self.elapsed_time += dt
        if self.elapsed_time >= self.time_limit:
            self.is_running = False
            print("Time Up!")
            return
            
        # ロボットの位置取得
        robot_pos = self.robot.get_pos()
        if robot_pos is None:
            return
            
        # Tensor -> Numpy (CPU)
        if hasattr(robot_pos, 'cpu'):
            robot_pos = robot_pos.cpu().numpy()
            
        # コイン獲得判定
        for spot in self.spots:
            if spot['collected']:
                continue
                
            # 距離判定 (XY平面)
            dist = np.sqrt((robot_pos[0] - spot['pos'][0])**2 + (robot_pos[1] - spot['pos'][1])**2)
            
            # 高さ判定 (Z) - 同じTierにいるか
            z_diff = abs(robot_pos[2] - spot['pos'][2])
            
            if dist < self.coin_radius and z_diff < 0.2:
                # 上段 (Tier 3) の場合は滞在時間が必要
                if spot['tier'] == 3:
                    spot['stay_timer'] += dt
                    if spot['stay_timer'] >= self.upper_tier_stay_time:
                        self._collect_spot(spot)
                else:
                    # 中・下段は即時獲得
                    self._collect_spot(spot)
            else:
                # エリアから出たらタイマーリセット
                if spot['tier'] == 3:
                    spot['stay_timer'] = 0.0
                    
    def _collect_spot(self, spot):
        """スポット獲得処理"""
        spot['collected'] = True
        points = 1 # 基本点
        if spot['tier'] == 3:
            points = 5 # 上段は高得点
        elif spot['tier'] == 2:
            points = 2
            
        self.score += points
        print(f"Spot Collected! ID={spot['id']}, Tier={spot['tier']}, Points={points}, Total Score={self.score}")
        
    def get_info(self):
        """表示用情報を返す"""
        return {
            'time': self.time_limit - self.elapsed_time,
            'score': self.score,
            'collected_count': sum(1 for s in self.spots if s['collected']),
            'total_spots': len(self.spots)
        }
