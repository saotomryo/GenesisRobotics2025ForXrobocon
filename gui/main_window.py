from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from gui.tabs.simulation_tab import SimulationTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.robot_config_tab import RobotConfigTab
from gui.tabs.evaluation_tab import EvaluationTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Genesis Robotics Control Center")
        self.setGeometry(100, 100, 1200, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout(central_widget)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Add tabs
        self.simulation_tab = SimulationTab()
        self.training_tab = TrainingTab()
        self.evaluation_tab = EvaluationTab()
        self.robot_config_tab = RobotConfigTab()
        
        self.tabs.addTab(self.simulation_tab, "シミュレーション")
        self.tabs.addTab(self.training_tab, "学習")
        self.tabs.addTab(self.evaluation_tab, "評価")
        self.tabs.addTab(self.robot_config_tab, "ロボット設定")
        
        # Settings tab placeholder
        settings_tab = QWidget()
        self.tabs.addTab(settings_tab, "設定")
