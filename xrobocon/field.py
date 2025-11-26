import genesis as gs
import numpy as np

class XRoboconField:
    """XROBOCON 3段フィールド生成クラス"""
    
    def __init__(self):
        # フィールド寸法 (単位: メートル)
        # 下段: φ3000mm, 高さ100mm
        # 中段: φ2000mm, 高さ250mm (下段からの相対高さ150mm)
        # 上段: φ1000mm, 高さ250mm (中段からの相対高さ0mm? いや、合計600mmなので...)
        # ルール確認:
        # 下段: φ3000mm, 高さ100mm
        # 中段: φ2000mm, 高さ250mm (合計350mm)
        # 上段: φ1000mm, 高さ250mm (合計600mm)
        
        self.tiers = [
            # 半径, 高さ(厚み), 中心Z座標 (床0基準)
            # 下段 (床から100mm) -> 中心Z = 0.05
            {'radius': 1.5, 'height': 0.1, 'z': 0.05, 'color': (0.8, 0.8, 0.8)},
            # 中段 (下段の上100mmから+250mm = 350mm) -> 中心Z = 0.1 + 0.125 = 0.225
            {'radius': 1.0, 'height': 0.25, 'z': 0.225, 'color': (0.6, 0.6, 1.0)},
            # 上段 (中段の上350mmから+250mm = 600mm) -> 中心Z = 0.35 + 0.125 = 0.475
            {'radius': 0.5, 'height': 0.25, 'z': 0.475, 'color': (1.0, 0.6, 0.6)},
        ]

    def build(self, scene):
        """シーンにフィールドエンティティを追加"""
        entities = []
        
        for i, tier in enumerate(self.tiers):
            # 円筒形の土俵を追加（静的オブジェクトとして固定）
            entity = scene.add_entity(
                gs.morphs.Cylinder(
                    pos=(0, 0, tier['z']),
                    height=tier['height'],
                    radius=tier['radius'],
                    fixed=True,  # 静的オブジェクト
                ),
                material=gs.materials.Rigid(
                    friction=0.8,
                )
            )
            entities.append(entity)
            print(f"Tier {i+1} created: Radius={tier['radius']}m, Height={tier['height']}m, Z={tier['z']}m")
            
        # スロープの追加 (螺旋状・接線方向・ソリッド風)
        # 戦略:
        # BoxはX軸方向を長手(Length)とする: size=(Length, Width, Thickness)
        # 配置:
        # 1. Ramp 1: Ground -> Tier 1 (R=1.5). 接点角度 0度.
        #    End(Top): (1.5, 0, 0.1). Tangent Dir: 90 deg (Y+).
        # 2. Ramp 2: Tier 1 -> Tier 2 (R=1.0). 接点角度 120度.
        #    End(Top): (1.0*cos120, 1.0*sin120, 0.35). Tangent Dir: 210 deg.
        # 3. Ramp 3: Tier 2 -> Tier 3 (R=0.5). 接点角度 240度.
        #    End(Top): (0.5*cos240, 0.5*sin240, 0.6). Tangent Dir: 330 deg.
        
        ramp_width = 0.8
        ramp_thickness = 0.5 # 厚めにして埋める
        
        def create_ramp(scene, name, start_h, end_h, target_r, target_angle_deg, length):
            # 目標地点 (円周上の接点)
            rad = np.radians(target_angle_deg)
            end_x = target_r * np.cos(rad)
            end_y = target_r * np.sin(rad)
            
            # 接線方向 (反時計回りなら +90度)
            tangent_deg = target_angle_deg + 90
            tangent_rad = np.radians(tangent_deg)
            dir_x = np.cos(tangent_rad)
            dir_y = np.sin(tangent_rad)
            
            # 傾斜角
            dh = end_h - start_h
            slope_rad = np.arcsin(dh / length)
            slope_deg = np.degrees(slope_rad)
            
            # 中心位置 (XY)
            # Endから接線方向逆向きに Length/2 戻る
            center_x = end_x - (dir_x * length / 2)
            center_y = end_y - (dir_y * length / 2)
            
            # 中心位置 (Z)
            # Z_pos = (H_top + H_bottom)/2 - (Thickness/2)*cos(slope)
            center_z = (start_h + end_h) / 2 - (ramp_thickness / 2) * np.cos(slope_rad)
            
            # Euler角 (Genesis: XYZ順と仮定)
            # 1. Y軸回転でSlopeをつける (X軸正方向を持ち上げるには -回転)
            #    Right-hand rule around Y: Z -> X. Positive rot moves X to -Z (down).
            #    So to move X up (Z+), we need Negative Y rotation.
            #    euler_y = -slope_deg
            # 2. Z軸回転で向きを合わせる
            #    euler_z = tangent_deg
            
            # Box作成 (X軸長手)
            ramp = scene.add_entity(
                gs.morphs.Box(
                    pos=(center_x, center_y, center_z),
                    size=(length, ramp_width, ramp_thickness),
                    euler=(0, -slope_deg, tangent_deg),
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=1.0)
            )
            return ramp

        # Ramp 1: Ground(0.0) -> Tier 1(0.1). Target R=1.5, Angle=0.
        # Length=1.5 (勾配緩やか)
        entities.append(create_ramp(scene, "ramp1", 0.0, 0.1, 1.5, 0, 1.5))
        
        # Ramp 2: Tier 1(0.1) -> Tier 2(0.35). Target R=1.0, Angle=120.
        # Length=1.0 (短くしてTier 1内に収める)
        # Start R approx 1.41 < 1.5
        entities.append(create_ramp(scene, "ramp2", 0.1, 0.35, 1.0, 120, 1.0))
        
        # Ramp 3: Tier 2(0.35) -> Tier 3(0.6). Target R=0.5, Angle=240.
        # Length=1.0 (短くしてTier 2内に収める)
        entities.append(create_ramp(scene, "ramp3", 0.35, 0.6, 0.5, 240, 1.0))
            
        return entities
