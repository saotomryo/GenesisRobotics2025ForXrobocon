from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QComboBox, QLabel, QGroupBox,
                               QSpinBox, QTextEdit, QProgressBar)
from PySide6.QtCore import QTimer
import sys
import os
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from core.robot_config_manager import RobotConfigManager

class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        
        self.process = None
        self.is_evaluating = False
        self.config_manager = RobotConfigManager()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Config group
        config_group = QGroupBox("評価設定")
        config_layout = QVBoxLayout()
        
        # Robot selection
        robot_layout = QHBoxLayout()
        robot_layout.addWidget(QLabel("ロボット:"))
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(["tristar", "tristar_large", "rocker_bogie", "rocker_bogie_large"])
        self.robot_combo.currentTextChanged.connect(self.update_model_list)
        robot_layout.addWidget(self.robot_combo)
        robot_layout.addStretch()
        config_layout.addLayout(robot_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("モデル:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        config_layout.addLayout(model_layout)
        
        # Environment selection
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("環境:"))
        self.env_combo = QComboBox()
        self.env_combo.addItems(["flat", "step", "step_hard"])
        self.env_combo.setCurrentText("step")
        env_layout.addWidget(self.env_combo)
        env_layout.addStretch()
        config_layout.addLayout(env_layout)
        
        # Number of episodes
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("評価エピソード数:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setMinimum(1)
        self.episodes_spin.setMaximum(1000)
        self.episodes_spin.setValue(10)
        episodes_layout.addWidget(self.episodes_spin)
        episodes_layout.addStretch()
        config_layout.addLayout(episodes_layout)
        
        # Render option
        from PySide6.QtWidgets import QCheckBox
        render_layout = QHBoxLayout()
        self.render_check = QCheckBox("シミュレーター表示")
        self.render_check.setChecked(False)  # デフォルトはオフ
        render_layout.addWidget(self.render_check)
        render_layout.addStretch()
        config_layout.addLayout(render_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.eval_btn = QPushButton("評価開始")
        self.eval_btn.clicked.connect(self.start_evaluation)
        button_layout.addWidget(self.eval_btn)
        
        self.save_btn = QPushButton("結果を保存")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)  # 初期状態では無効
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results display
        results_group = QGroupBox("評価結果")
        results_layout = QVBoxLayout()
        
        # Text results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.setLayout(layout)
        
        # Load models for first robot
        self.update_model_list(self.robot_combo.currentText())
    
    def update_model_list(self, robot_name):
        """ロボットに応じてモデルリストを更新"""
        self.model_combo.clear()
        
        # Load robot config
        config = self.config_manager.load_config(robot_name)
        trained_models = config.get('trained_models', [])
        
        # Convert old format if needed
        if isinstance(trained_models, dict):
            trained_models = list(trained_models.values())
        
        if not trained_models:
            self.model_combo.addItem("モデルなし", None)
            self.eval_btn.setEnabled(False)
        else:
            self.eval_btn.setEnabled(True)
            for model_path in trained_models:
                # Display only filename
                model_name = os.path.basename(model_path)
                self.model_combo.addItem(model_name, model_path)
    
    def start_evaluation(self):
        """評価を開始"""
        try:
            robot_type = self.robot_combo.currentText()
            model_data = self.model_combo.currentData()
            env_type = self.env_combo.currentText()
            num_episodes = self.episodes_spin.value()
            
            if not model_data:
                self.results_text.append("エラー: モデルが選択されていません\n")
                return
            
            self.is_evaluating = True
            self.eval_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.results_text.clear()
            self.results_text.append("評価を開始しています...\n")
            
            # Build command
            script_path = os.path.join(os.path.dirname(__file__), '../../scripts/evaluate_model.py')
            project_root = os.path.join(os.path.dirname(__file__), '../..')
            project_root = os.path.abspath(project_root)
            
            model_path = os.path.join(project_root, model_data)
            
            cmd = [
                sys.executable, script_path,
                '--model', model_path,
                '--env', env_type,
                '--robot', robot_type,
                '--episodes', str(num_episodes)
            ]
            
            # Add --render flag if checkbox is checked
            if self.render_check.isChecked():
                cmd.append('--render')
            
            self.results_text.append(f"コマンド: {' '.join(cmd)}\n\n")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for completion and read output
            output, _ = self.process.communicate()
            
            self.results_text.append(output)
            
            # 結果を保存できるように保存ボタンを有効化
            self.last_evaluation_output = output
            self.last_evaluation_config = {
                'robot': robot_type,
                'model': os.path.basename(model_data),
                'env': env_type,
                'episodes': num_episodes
            }
            self.save_btn.setEnabled(True)
            
            self.is_evaluating = False
            self.eval_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            import traceback
            self.results_text.append(f"\nエラー: {str(e)}\n{traceback.format_exc()}\n")
            self.is_evaluating = False
            self.eval_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def save_results(self):
        """評価結果をファイルに保存"""
        from PySide6.QtWidgets import QFileDialog
        from datetime import datetime
        
        if not hasattr(self, 'last_evaluation_output'):
            return
        
        # デフォルトのファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = self.last_evaluation_config
        default_filename = f"eval_{config['robot']}_{config['env']}_{timestamp}.txt"
        
        # ファイル保存ダイアログ
        project_root = os.path.join(os.path.dirname(__file__), '../..')
        project_root = os.path.abspath(project_root)
        results_dir = os.path.join(project_root, 'evaluation_results')
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(results_dir, exist_ok=True)
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "評価結果を保存",
            os.path.join(results_dir, default_filename),
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    # ヘッダー情報
                    f.write("=" * 70 + "\n")
                    f.write("評価結果レポート\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"ロボット: {config['robot']}\n")
                    f.write(f"モデル: {config['model']}\n")
                    f.write(f"環境: {config['env']}\n")
                    f.write(f"評価エピソード数: {config['episodes']}\n")
                    f.write("\n" + "=" * 70 + "\n\n")
                    
                    # 評価結果
                    f.write(self.last_evaluation_output)
                
                self.results_text.append(f"\n結果を保存しました: {filename}\n")
                
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "エラー",
                    f"ファイルの保存に失敗しました: {str(e)}"
                )
