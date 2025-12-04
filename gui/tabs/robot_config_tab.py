from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLineEdit, QLabel, QGroupBox,
                               QComboBox, QFormLayout, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from xrobocon.robot_configs import ROBOT_CONFIGS, get_robot_config
from core.robot_config_manager import RobotConfigManager

class RobotConfigTab(QWidget):
    def __init__(self):
        super().__init__()
        
        self.current_robot = None
        self.config_inputs = {}
        self.config_manager = RobotConfigManager()
        
        # 初期化：デフォルト設定ファイルを作成
        self.config_manager.initialize_all_configs()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Robot selection
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("ロボット選択:"))
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(list(ROBOT_CONFIGS.keys()))
        self.robot_combo.currentTextChanged.connect(self.load_robot_config)
        select_layout.addWidget(self.robot_combo)
        select_layout.addStretch()
        layout.addLayout(select_layout)
        
        # Scroll area for config
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.config_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("設定を保存")
        self.save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("リセット")
        self.reset_btn.clicked.connect(self.reset_config)
        button_layout.addWidget(self.reset_btn)
        
        self.export_btn = QPushButton("設定ファイルを開く")
        self.export_btn.clicked.connect(self.open_config_file)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Load first robot
        if self.robot_combo.count() > 0:
            self.load_robot_config(self.robot_combo.currentText())
    
    def load_robot_config(self, robot_name):
        """ロボット設定を読み込んで表示"""
        self.current_robot = robot_name
        config = self.config_manager.load_config(robot_name)
        
        # Clear existing inputs
        while self.config_layout.count():
            child = self.config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.config_inputs = {}
        
        # Model files section
        model_group = QGroupBox("モデルファイル")
        model_layout = QVBoxLayout()
        
        # Robot model file
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("ロボットモデル:"))
        model_file = config.get('model_file', '')
        model_input = QLineEdit(str(model_file))
        self.config_inputs['model_file'] = model_input
        model_file_layout.addWidget(model_input)
        model_layout.addLayout(model_file_layout)
        
        # Trained models list
        model_layout.addWidget(QLabel("学習済みモデル:"))
        
        # List widget for trained models
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        self.trained_models_list = QListWidget()
        trained_models = config.get('trained_models', [])
        
        # Convert old format to new format if needed
        if isinstance(trained_models, dict):
            trained_models = list(trained_models.values())
            
        for model_path in trained_models:
            self.trained_models_list.addItem(model_path)
        
        model_layout.addWidget(self.trained_models_list)
        
        # Buttons for model management
        model_btn_layout = QHBoxLayout()
        add_model_btn = QPushButton("モデルを追加")
        add_model_btn.clicked.connect(self.add_trained_model)
        model_btn_layout.addWidget(add_model_btn)
        
        remove_model_btn = QPushButton("選択したモデルを削除")
        remove_model_btn.clicked.connect(self.remove_trained_model)
        model_btn_layout.addWidget(remove_model_btn)
        model_btn_layout.addStretch()
        model_layout.addLayout(model_btn_layout)
        
        model_group.setLayout(model_layout)
        self.config_layout.addWidget(model_group)
        
        # Physics parameters
        physics_group = QGroupBox("物理パラメータ")
        physics_form = QFormLayout()
        
        physics = config.get('physics', {})
        for key, value in physics.items():
            input_field = QLineEdit(str(value))
            self.config_inputs[f'physics.{key}'] = input_field
            physics_form.addRow(QLabel(key + ":"), input_field)
        
        physics_group.setLayout(physics_form)
        self.config_layout.addWidget(physics_group)
        
        # Control parameters
        control_group = QGroupBox("制御パラメータ")
        control_form = QFormLayout()
        
        control = config.get('control', {})
        for key, value in control.items():
            input_field = QLineEdit(str(value))
            self.config_inputs[f'control.{key}'] = input_field
            control_form.addRow(QLabel(key + ":"), input_field)
        
        control_group.setLayout(control_form)
        self.config_layout.addWidget(control_group)
        
        # Reward parameters
        reward_group = QGroupBox("報酬パラメータ")
        reward_form = QFormLayout()
        
        reward = config.get('reward_params', {})
        for key, value in reward.items():
            input_field = QLineEdit(str(value))
            self.config_inputs[f'reward_params.{key}'] = input_field
            reward_form.addRow(QLabel(key + ":"), input_field)
        
        reward_group.setLayout(reward_form)
        self.config_layout.addWidget(reward_group)
        
        self.config_layout.addStretch()
    
    def add_trained_model(self):
        """学習済みモデルを追加"""
        from PySide6.QtWidgets import QFileDialog
        
        project_root = os.path.join(os.path.dirname(__file__), '../..')
        project_root = os.path.abspath(project_root)
        models_dir = os.path.join(project_root, 'models')
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "学習済みモデルを選択",
            models_dir,
            "Model Files (*.zip)"
        )
        
        if filename:
            # 相対パスに変換
            try:
                rel_path = os.path.relpath(filename, project_root)
                self.trained_models_list.addItem(rel_path)
            except:
                self.trained_models_list.addItem(filename)
    
    def remove_trained_model(self):
        """選択した学習済みモデルを削除"""
        current_item = self.trained_models_list.currentItem()
        if current_item:
            self.trained_models_list.takeItem(self.trained_models_list.row(current_item))
    
    def save_config(self):
        """設定をJSONファイルに保存"""
        try:
            # Collect values
            new_config = {
                'robot_name': self.current_robot,
                'model_file': '',
                'trained_models': [],
                'physics': {},
                'control': {},
                'reward_params': {}
            }
            
            # Get model file
            if 'model_file' in self.config_inputs:
                new_config['model_file'] = self.config_inputs['model_file'].text()
            
            # Get trained models from list
            for i in range(self.trained_models_list.count()):
                item = self.trained_models_list.item(i)
                new_config['trained_models'].append(item.text())
            
            # Get other parameters
            for key, input_field in self.config_inputs.items():
                if key == 'model_file':
                    continue
                    
                value_str = input_field.text()
                parts = key.split('.')
                section = parts[0]
                param = parts[1]
                
                # Try to convert to appropriate type
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str
                
                new_config[section][param] = value
            
            # Save to JSON file
            self.config_manager.save_config(self.current_robot, new_config)
            
            QMessageBox.information(
                self, 
                "成功", 
                f"{self.current_robot}の設定を保存しました。\n"
                f"ファイル: {self.config_manager.get_config_path(self.current_robot)}"
            )
            
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, 
                "エラー", 
                f"設定の保存に失敗しました: {str(e)}\n\n{traceback.format_exc()}"
            )
    
    def reset_config(self):
        """設定をリセット"""
        if self.current_robot:
            reply = QMessageBox.question(
                self,
                '確認',
                'デフォルト設定に戻しますか？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # デフォルト設定を取得して保存
                default_config = self.config_manager.get_default_config(self.current_robot)
                self.config_manager.save_config(self.current_robot, default_config)
                # 再読み込み
                self.load_robot_config(self.current_robot)
                QMessageBox.information(self, "完了", "デフォルト設定に戻しました。")
    
    def open_config_file(self):
        """設定ファイルをシステムのデフォルトエディタで開く"""
        import subprocess
        import platform
        
        config_path = self.config_manager.get_config_path(self.current_robot)
        
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', config_path])
            elif platform.system() == 'Windows':
                os.startfile(config_path)
            else:  # Linux
                subprocess.run(['xdg-open', config_path])
        except Exception as e:
            QMessageBox.warning(
                self,
                "エラー",
                f"ファイルを開けませんでした: {str(e)}\n\nパス: {config_path}"
            )
