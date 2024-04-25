from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt6.QtCore import Qt, QPoint, QTimer

from Interface.AbstractTab import AbstractTab
from main import EyesDetector, Camera, EyeDistances
import cv2


class CalibrationTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.screen_size = self.screen().size()
        self.point_size = 10
        self.half_point_size = self.point_size / 2
        self.screen_size.setHeight(self.screen_size.height() - 90)
        self.setWindowTitle("Calibaration Tab")
        self.points = [
            QPoint(self.half_point_size, self.half_point_size),
            QPoint(self.screen_size.width() / 2, self.screen_size.height() - self.half_point_size),
            QPoint(self.screen_size.width() - self.half_point_size, self.screen_size.height() / 2),
            QPoint(self.half_point_size, self.screen_size.height() - self.half_point_size),
            QPoint(self.screen_size.width() / 2, self.half_point_size),
            QPoint(self.screen_size.width() - self.half_point_size, self.screen_size.height() - self.half_point_size),
            QPoint(self.screen_size.width() / 2, self.screen_size.height() / 2),
            QPoint(self.screen_size.width() - self.half_point_size, self.half_point_size),
            QPoint(self.half_point_size, self.screen_size.height() / 2)
        ]
        self.counter = 0
        self.eye_detector = EyesDetector()
        self.camera = parent_class.camera
        self.eye_distances = EyeDistances()

    def paintEvent(self, event):
        if self.counter > 8:
            return
        painter = QPainter(self)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(12)
        painter.setPen(pen)
        painter.drawPoint(self.points[self.counter])

    def mousePressEvent(self, event):
        if self.counter > 8:
            return
        click_pos = event.position().toPoint()
        if (click_pos - self.points[self.counter]).manhattanLength() <= 10:
            print(f"point clicked at: {click_pos.x()}, {click_pos.y()}")
            self.counter += 1
            # if self.counter == 9:
            #     self.close()

            frame = self.camera.read_frame()
            hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = hvs
            results = self.eye_detector.get_face_mesh_results(frame)
            self.eye_detector.get_eyes_coordinates(results, frame, self.eye_distances)
            print(self.eye_distances.distance_percentage_x)
            self.update()

    def tab_selected(self):
        self.camera.start_capture()
