from PyQt6.QtGui import QPainter, QPen, QColor, QPalette
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel
from PyQt6.QtCore import Qt, QPoint
from main import EyesDetector, Camera, EyeDistances
import cv2


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 0))
        self.setPalette(palette)
