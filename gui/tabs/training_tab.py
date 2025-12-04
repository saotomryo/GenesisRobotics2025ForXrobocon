from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLineEdit, QLabel, QTextEdit, 
                               QGroupBox, QComboBox, QFileDialog)
from PySide6.QtCore import QTimer, QProcess
import sys
import os
import subprocess
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from core.robot_config_manager import RobotConfigManager

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        
        self.process = None
        self.is_training = False
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_process_output)
        self.config_manager = RobotConfigManager()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Config group
        config_group = QGroupBox("学習設定")
        config_layout = QVBoxLayout()
        
        # Steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("学習ステップ数:"))
        self.steps_input = QLineEdit("10000")
        steps_layout.addWidget(self.steps_input)
        steps_layout.addStretch()
        config_layout.addLayout(steps_layout)
        
        # Environment
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("環境:"))
        self.env_combo = QComboBox()
        self.env_combo.addItems(["flat", "step", "step_hard"])
        self.env_combo.setCurrentText("step")
        env_layout.addWidget(self.env_combo)
        env_layout.addStretch()
        config_layout.addLayout(env_layout)
        
        # Robot
        robot_layout = QHBoxLayout()
        robot_layout.addWidget(QLabel("ロボット:"))
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(["tristar", "tristar_large", "rocker_bogie", "rocker_bogie_large"])
        self.robot_combo.currentTextChanged.connect(self.update_base_model_list)
        robot_layout.addWidget(self.robot_combo)
        robot_layout.addStretch()
        config_layout.addLayout(robot_layout)
        
        # Base model
        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel("ベースモデル:"))
        self.base_combo = QComboBox()
        base_layout.addWidget(self.base_combo)
        base_layout.addStretch()
        config_layout.addLayout(base_layout)
        
        # Save name
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("保存名:"))
        self.save_input = QLineEdit("trained_model")
        save_layout.addWidget(self.save_input)
        save_layout.addStretch()
        config_layout.addLayout(save_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Start button
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("学習開始")
        self.start_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Progress log
        log_group = QGroupBox("学習ログ")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        
        # Load models for first robot
        self.update_base_model_list(self.robot_combo.currentText())
    
    def update_base_model_list(self, robot_name):
        """ロボットに応じてベースモデルリストを更新"""
        self.base_combo.clear()
        self.base_combo.addItem("なし（新規学習）", None)
        
        # Load robot config
        config = self.config_manager.load_config(robot_name)
        trained_models = config.get('trained_models', [])
        
        # Convert old format if needed
        if isinstance(trained_models, dict):
            trained_models = list(trained_models.values())
        
        for model_path in trained_models:
            # Display only filename
            model_name = os.path.basename(model_path)
            self.base_combo.addItem(model_name, model_path)
            
    def start_training(self):
        try:
            steps = int(self.steps_input.text())
            env_type = self.env_combo.currentText()
            robot_type = self.robot_combo.currentText()
            base_model_data = self.base_combo.currentData()
            save_name = self.save_input.text().strip()
            
            # Build command - use appropriate script based on environment
            if env_type == "step_hard":
                script_path = os.path.join(os.path.dirname(__file__), '../../scripts/train_step_hard_loop.py')
            elif env_type == "step":
                script_path = os.path.join(os.path.dirname(__file__), '../../scripts/train_step_loop.py')
            else:  # flat
                script_path = os.path.join(os.path.dirname(__file__), '../../scripts/train_rl.py')
            
            project_root = os.path.join(os.path.dirname(__file__), '../..')
            project_root = os.path.abspath(project_root)
            
            # Build command based on environment
            if env_type == "flat":
                # train_rl.py uses different arguments
                cmd = [
                    sys.executable, script_path,
                    '--train',
                    '--steps', str(steps),
                    '--env', 'flat',
                    '--robot', robot_type
                ]
                if save_name:
                    cmd.extend(['--save_name', save_name])
            else:
                # train_step_loop.py and train_step_hard_loop.py
                cmd = [
                    sys.executable, script_path,
                    '--steps', str(steps),
                    '--chunk', '10000',  # 10000ステップごとにGenesisを再起動
                    '--robot', robot_type
                ]
                if save_name:
                    cmd.extend(['--save_name', save_name])
            
            self.log_text.append(f"学習を開始しています...\n")
            self.log_text.append(f"総ステップ数: {steps}\n")
            self.log_text.append(f"ロボット: {robot_type}, 環境: {env_type}\n")
            if save_name:
                self.log_text.append(f"保存名: {save_name}.zip\n")
            else:
                if env_type == "flat":
                    self.log_text.append(f"保存名: xrobocon_ppo_{robot_type}_flat.zip (デフォルト)\n")
                elif env_type == "step_hard":
                    self.log_text.append(f"保存名: xrobocon_ppo_{robot_type}_step_hard.zip (デフォルト)\n")
                else:
                    self.log_text.append(f"保存名: xrobocon_ppo_{robot_type}_step.zip (デフォルト)\n")
            if base_model_data:
                self.log_text.append(f"ベースモデル: {os.path.basename(base_model_data)}\n")
                if env_type != "flat":
                    self.log_text.append(f"注意: train_step_loopは既存モデルから自動的に再開します\n")
            self.log_text.append(f"コマンド: {' '.join(cmd)}\n\n")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.is_training = True
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor output
            self.check_timer.start(100) # Check every 100 ms
            
        except Exception as e:
            self.log_text.append(f"エラー: {str(e)}\n{traceback.format_exc()}\n")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.is_training = False

    def check_process_output(self):
        if self.process and self.process.poll() is None: # Process is still running
            output = self.process.stdout.readline()
            if output:
                self.log_text.append(output.strip())
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )
        elif self.process and self.process.poll() is not None: # Process has finished
            self.check_timer.stop()
            # Read any remaining output
            for output in self.process.stdout.readlines():
                if output:
                    self.log_text.append(output.strip())
            
            return_code = self.process.returncode
            if return_code == 0:
                self.log_text.append("\n学習が正常に完了しました。\n")
            else:
                self.log_text.append(f"\n学習がエラーで終了しました。終了コード: {return_code}\n")
            
            self.training_finished()

    def stop_training(self):
        if self.process and self.process.poll() is None:
            self.log_text.append("\n学習を停止しています...\n")
            self.process.terminate() # or .kill() for a more forceful stop
            self.process.wait() # Wait for the process to terminate
            self.log_text.append("学習が停止されました。\n")
            self.training_finished()
        else:
            self.log_text.append("実行中の学習プロセスはありません。\n")

    def training_finished(self):
        self.check_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_training = False
        self.process = None # Clear the process reference
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
