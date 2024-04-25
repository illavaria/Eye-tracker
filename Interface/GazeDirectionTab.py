from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QHBoxLayout
from PyQt6.QtCore import QTimer
from Interface.AbstractTab import AbstractTab
import cv2


class GazeDirectionTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.camera = parent_class.camera
        self.layout = QHBoxLayout(self)
        self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        frame = self.camera.read_frame()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)

            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)

    def tab_selected(self):
        self.camera.start_capture()
        self.timer.start(30)

