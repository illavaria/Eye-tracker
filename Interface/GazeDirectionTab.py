from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QRadioButton, QMessageBox
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
        self.exception_label = QLabel(self)
        self.radio_button = QRadioButton(self)

        self.eye_detector = parent_class.eye_detector
        self.eyes_recognizer = parent_class.eyes_recognizer
        self.eye_distances = EyeDistances()
        self.gaze_prediction = GazeDirectionPrediction()
        self.counter = 0

        empty_label = QLabel(self)

        vbox = QVBoxLayout(self)
        # vbox.setSpacing(0)
        # vbox.setContentsMargins(0, 0, 0, 0)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exception_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label.setFixedWidth(150)
        self.direction_label.setFixedWidth(150)
        self.exception_label.setFixedWidth(150)

        hbox = QHBoxLayout(self)
        hbox.addWidget(self.radio_button, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox.addLayout(hbox)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.direction_label)
        vbox.addWidget(self.exception_label)
        for i in range(12):
            vbox.addWidget(empty_label)

        self.layout.addWidget(self.image_label)
        self.layout.addLayout(vbox)
        self.setLayout(self.layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        frame = self.camera.read_frame()
        if frame is not None:
            hvs = ImageEdit.split_image(frame)
            image = ImageEdit.image_to_RGB(frame)
            try:
                if self.counter % 5 == 0:
                    results = self.eye_detector.get_face_mesh_results(hvs)
                    self.eye_detector.get_eyes_coordinates(results, hvs, self.eye_distances)
                self.eyes_recognizer.get_eyes_coordinates(hvs, self.eye_distances)
            except Exception as e:
                self.exception_label.setText("Don't hide the face")
                self.direction_label.setText('')
                self.counter = 4
            else:
                self.exception_label.setText('')
                direction = self.gaze_prediction.predict(self.eye_distances)
                self.direction_label.setText(direction.value)
                if self.radio_button.isChecked():
                    image = self.eye_distances.draw(image)

            image = ImageEdit.flip_image(image)
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
            self.counter += 1

    def tab_selected(self):
        if not self.eyes_recognizer.is_loaded:
            QMessageBox.warning(self, 'No loaded model', "There aren't any downloaded models on the device. Please go "
                                                         "to settings tab and download a model.")
            return
        self.radio_button.setText('Draw dots')
        self.text_label.setText('Gaze direction:')
        self.direction_label.setText('Forward')
        self.camera.start_capture()
        frame = self.camera.read_frame()
        if frame is not None:
            hvs = ImageEdit.split_image(frame)
            try:
                results = self.eye_detector.get_face_mesh_results(hvs)
                self.eye_detector.get_eyes_coordinates(results, hvs, self.eye_distances)
            except Exception as e:
                self.exception_label.setText("Don't hide the face")

        self.timer.start(30)
        self.counter = 0

