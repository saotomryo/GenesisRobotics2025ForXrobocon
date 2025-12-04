from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QComboBox, QLabel, QGroupBox)
from PySide6.QtCore import QTimer
import sys
import os
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from core.robot_config_manager import RobotConfigManager

class SimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        
        self.process = None
        self.is_running = False
        self.model_path = None
        self.config_manager = RobotConfigManager()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control group
        control_group = QGroupBox("シミュレーション設定")
        control_layout = QVBoxLayout()
        
        # Environment selection
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("環境:"))
        self.env_combo = QComboBox()
        self.env_combo.addItems(["flat", "step"])
        self.env_combo.setCurrentText("step")
        env_layout.addWidget(self.env_combo)
        env_layout.addStretch()
        control_layout.addLayout(env_layout)
        
        # Robot selection
        robot_layout = QHBoxLayout()
        robot_layout.addWidget(QLabel("ロボット:"))
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(["tristar", "tristar_large", "rocker_bogie", "rocker_bogie_large"])
        self.robot_combo.currentTextChanged.connect(self.update_model_list)
        robot_layout.addWidget(self.robot_combo)
        robot_layout.addStretch()
        control_layout.addLayout(robot_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("モデル:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        control_layout.addLayout(model_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("シミュレーション開始")
        self.start_btn.clicked.connect(self.toggle_simulation)
        button_layout.addWidget(self.start_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("ステータス: 準備完了")
        layout.addWidget(self.status_label)
        
        layout.addStretch()        
        self.setLayout(layout)
        
        # Load models for first robot
        self.update_model_list(self.robot_combo.currentText())
    
    def update_model_list(self, robot_name):
        """ロボットに応じてモデルリストを更新"""
        self.model_combo.clear()
        self.model_combo.addItem("なし", None)
        
        # Load robot config
        config = self.config_manager.load_config(robot_name)
        trained_models = config.get('trained_models', [])
        
        # Convert old format if needed
        if isinstance(trained_models, dict):
            trained_models = list(trained_models.values())
        
        for model_path in trained_models:
            # Display only filename
            model_name = os.path.basename(model_path)
            self.model_combo.addItem(model_name, model_path)  # Store full path as data
    
    def toggle_simulation(self):
        if not self.is_running:
            self.start_simulation()
        else:
            self.stop_simulation()
            
    def start_simulation(self):
        try:
            # Get selected model
            robot_type = self.robot_combo.currentText()
            model_data = self.model_combo.currentData()
            
            self.is_running = True
            self.start_btn.setText("停止")
            self.start_btn.setStyleSheet("background-color: #ff4444;")
            self.status_label.setText("ステータス: シミュレーター起動中...")
            
            # Build command
            env_type = self.env_combo.currentText()
            
            # Use visualize_trained_model.py script
            script_path = os.path.join(os.path.dirname(__file__), '../../scripts/visualize_trained_model.py')
            project_root = os.path.join(os.path.dirname(__file__), '../..')
            project_root = os.path.abspath(project_root)
            
            cmd = [sys.executable, script_path, '--env', env_type, '--robot', robot_type]
            
            if model_data:  # model_data is None for "なし"
                model_path = os.path.join(project_root, model_data)
                cmd.extend(['--model', model_path])
            
            # Start process with project root as cwd
            self.process = subprocess.Popen(cmd, cwd=project_root)
            self.status_label.setText(f"ステータス: 実行中 (PID: {self.process.pid})")
            
        except Exception as e:
            import traceback
            error_msg = f"エラー: {str(e)}\n{traceback.format_exc()}"
            self.status_label.setText(f"エラー: {str(e)}")
            print(error_msg)
            self.is_running = False
            self.start_btn.setText("シミュレーション開始")
            self.start_btn.setStyleSheet("")
            
    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
        
        self.is_running = False
        self.start_btn.setText("シミュレーション開始")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("ステータス: 停止")

