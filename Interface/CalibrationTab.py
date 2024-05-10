from PyQt6.QtGui import QPainter, QPen, QColor, QPalette
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QPoint, QTimer

from Interface.AbstractTab import AbstractTab
from main import EyesDetector, Camera, EyeDistances, EyesRecognizer, ImageEdit
import cv2


class CalibrationTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.screen_size = parent_class.screen_size
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
        self.counter = -1
        self.eye_detector = parent_class.eye_detector
        self.eyes_recognizer = parent_class.eyes_recognizer
        self.camera = parent_class.camera
        self.eye_distances = EyeDistances()
        self.screen_params_list = [parent_class.cheating_detection.lct, parent_class.cheating_detection.bm,
                                   parent_class.cheating_detection.rm, parent_class.cheating_detection.lcb,
                                   parent_class.cheating_detection.tm, parent_class.cheating_detection.rcb,
                                   parent_class.cheating_detection.middle, parent_class.cheating_detection.rct,
                                   parent_class.cheating_detection.lm]

    def paintEvent(self, event):
        if self.counter > 8 or self.counter == -1:
            return
        painter = QPainter(self)
        background_color = self.palette().color(self.backgroundRole())
        if background_color.lightness() > 127:
            pen = QPen(QColor(0, 0, 0))
        else:
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

            frame = self.camera.read_frame()
            frame = ImageEdit.split_image(frame)
            try:
                results = self.eye_detector.get_face_mesh_results(frame)
                self.eye_detector.get_eyes_coordinates(results, frame, self.screen_params_list[self.counter])
                self.eyes_recognizer.get_eyes_coordinates(frame, self.screen_params_list[self.counter])
            except Exception as e:
                QMessageBox.critical(self, "Face detection error", "Don't hide the face")
                return

            self.counter += 1
            if self.counter == 9:
                self.parent_class.calibration_taken = True
                self.parent_class.cheating_detection.calculate_borders()
                print(self.parent_class.cheating_detection.left_border_percentage_x_avg)
                print(self.parent_class.cheating_detection.right_border_percentage_x_avg)
                print(self.parent_class.cheating_detection.left_border_angle_diff_avg)
                print(self.parent_class.cheating_detection.right_border_angle_diff_avg)
                print(self.parent_class.cheating_detection.up_border_angle_avg)
                print(self.parent_class.cheating_detection.down_border_angle_avg)
                QMessageBox.information(self, "Calibration over", "Calibration is finished, you can go to other tabs")
            self.update()

    def tab_selected(self):
        if not self.eyes_recognizer.is_loaded:
            QMessageBox.warning(self, 'No loaded model', "There aren't any downloaded models on the device. Please go "
                                                         "to settings tab and download a model.")
            self.counter = -1
            return
        self.camera.start_capture()
        self.counter = 0
