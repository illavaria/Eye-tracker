from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QRadioButton
from PyQt6.QtCore import QTimer, Qt
from Interface.AbstractTab import AbstractTab
import cv2

from main import EyesDetector, EyeDistances, ImageEdit, GazeDirectionPrediction, EyesRecognizer


class GazeDirectionTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.camera = parent_class.camera
        self.layout = QHBoxLayout(self)
        self.image_label = QLabel(self)
        self.text_label = QLabel(self)
        self.direction_label = QLabel(self)
        self.radio_button = QRadioButton('Draw Dots', self)

        self.eye_detector = EyesDetector()
        self.eyes_recognizer = EyesRecognizer(
            "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_left_my_dataset_fixed_photos/epoch_400.pth",
            "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_right_my_dataset_fixed_photos/epoch_400.pth")
        self.eye_distances = EyeDistances()
        self.gaze_prediction = GazeDirectionPrediction()
        self.counter = 0

        empty_label = QLabel(self)

        vbox = QVBoxLayout(self)
        # vbox.setSpacing(0)
        # vbox.setContentsMargins(0, 0, 0, 0)
        self.text_label.setText('Gaze direction:')
        self.direction_label.setText('Forward')
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vbox.addWidget(self.radio_button)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.direction_label)
        for i in range(12):
            vbox.addWidget(empty_label)
        # self.image_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.image_label)
        self.layout.addLayout(vbox)
        self.setLayout(self.layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        frame = self.camera.read_frame()
        if frame is not None:
            hvs = ImageEdit.split_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)

            pixmap = QPixmap.fromImage(image)
            if self.counter % 5 == 0:
                results = self.eye_detector.get_face_mesh_results(hvs)
                self.eye_detector.get_eyes_coordinates(results, hvs, self.eye_distances)
            self.eyes_recognizer.get_eyes_coordinates(hvs, self.eye_distances)
            direction = self.gaze_prediction.predict(self.eye_distances)
            self.direction_label.setText(direction.value)
            self.image_label.setPixmap(pixmap)
            self.counter += 1

    def tab_selected(self):
        self.camera.start_capture()
        frame = self.camera.read_frame()
        if frame is not None:
            hvs = ImageEdit.split_image(frame)
            results = self.eye_detector.get_face_mesh_results(hvs)
            self.eye_detector.get_eyes_coordinates(results, hvs, self.eye_distances)
        self.timer.start(30)
        self.counter = 0

