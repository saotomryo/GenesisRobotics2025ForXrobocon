import sys
import os

# Ensure root directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Genesis Robotics Control Center")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
