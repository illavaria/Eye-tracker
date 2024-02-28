import cv2
import mediapipe as mp
import numpy as np


class Eye:
    center_x, center_y = 0, 0
    inside_x, inside_y = 0, 0
    outside_x, outside_y = 0, 0
    top_x, top_y = 0, 0
    bottom_x, bottom_y = 0, 0
    
    def draw(self, frame):
        cv2.drawMarker(frame, (self.center_x, self.center_y), (0, 255, 0), markerSize=5)
        cv2.drawMarker(frame, (self.inside_x, self.inside_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.outside_x, self.outside_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.top_x, self.top_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.bottom_x, self.bottom_y), (255, 0, 255), markerSize=5)
        return frame
    

class EyesDetector:
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                refine_landmarks=True,
                                                max_num_faces=2,
                                                min_detection_confidence=0.5)
    left_eye = Eye()
    right_eye = Eye()

    def get_face_mesh_results(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_eyes_coordinates(self, results, frame):
        frame_h, frame_w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            # Iterate over all landmark
            for idx, lm in enumerate(face_landmarks.landmark):

                # Right iris center landmark
                if idx == 468:
                    self.right_eye.center_x, self.right_eye.center_y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Left iris center landmark
                if idx == 473:
                    self.left_eye.center_x, self.left_eye.center_y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Eye inside corner landmark
                if idx == 133:
                    self.right_eye.inside_x, self.right_eye.inside_y = int(lm.x * frame_w), int(lm.y * frame_h)
                if idx == 362:
                    self.left_eye.inside_x, self.left_eye.inside_y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Eye outside corner landmark
                if idx == 33:
                    self.right_eye.outside_x, self.right_eye.outside_y = int(lm.x * frame_w), int(lm.y * frame_h)
                if idx == 263: # for left eye 130 is like the corner of skin, 33 of purpil, 466 for right
                    self.left_eye.outside_x, self.left_eye.outside_y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Left/right Top eye landmark
                if idx == 159:
                    self.right_eye.top_x, self.right_eye.top_y = int(lm.x * frame_w), int(lm.y * frame_h)
                if idx == 386:
                    self.left_eye.top_x, self.left_eye.top_y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Left/right Bottom eye landmark
                if idx == 145:
                    self.right_eye.bottom_x, self.right_eye.bottom_y = int(lm.x * frame_w), int(lm.y * frame_h)
                if idx == 374:
                    self.left_eye.bottom_x, self.left_eye.bottom_y = int(lm.x * frame_w), int(lm.y * frame_h)

            # # Calculate left eye scores (x,y)
            # if (inside_x - outside_x) != 0:
            #     lx_score = (lcx - outside_x) / (inside_x - outside_x)
            #     if abs(lx_score - self.last_x) < .3:
            #         cv2.putText(frame, "x: {:.02}".format(lx_score), (lcx, lcy - 30), cv2.FONT_HERSHEY_SIMPLEX, .25,
            #                     (255, 255, 255), 1)
            #     last_x = lx_score
            # 
            # if (bottom - top) != 0:
            #     ly_score = (lcy - top) / (bottom - top)
            #     if abs(ly_score - self.last_y) < .3:
            #         cv2.putText(frame, "y: {:.02}".format(ly_score), (lcx, lcy - 20), cv2.FONT_HERSHEY_SIMPLEX, .25,
            #                     (255, 255, 255), 1)
            #     last_y = ly_score

class CalibrationData:
    # Left corner top
    lct_left_eye, lct_right_eye = Eye(), Eye()

    def __init__(self):
        cap = cv2.VideoCapture(0)
        eyeDetector = EyesDetector()

        while True:
            ret, frame = cap.read()
            results = eyeDetector.get_face_mesh_results(frame)
            if not results.multi_face_landmarks:
                continue
            eyeDetector.get_eyes_coordinates(results, frame)
            frame_right_eye = eyeDetector.right_eye.draw(frame)
            result_frame = eyeDetector.left_eye.draw(frame_right_eye)
            result_frame = cv2.flip(result_frame, 1)
            cv2.imshow('calibrate eyes', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                lct_left_eye, lct_right_eye = eyeDetector.left_eye, eyeDetector.right_eye
                cv2.imwrite('result.jpg', result_frame)
                cap.release()
                #cv2.destroyWindow('calibrate eyes')
                #cv2.destroyAllWindows()
                break




def main():
    calibrationData = CalibrationData()
    cap = cv2.VideoCapture(0)
    eyeDetector = EyesDetector()

    while True:
        ret, frame = cap.read()

        # To improve performance
        frame.flags.writeable = False

        results = eyeDetector.get_face_mesh_results(frame)

        # Make it writeable again
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            continue

        eyeDetector.get_eyes_coordinates(results, frame)
        frame_right_eye = eyeDetector.right_eye.draw(frame)
        result_frame = eyeDetector.left_eye.draw(frame_right_eye)

        result_frame = cv2.flip(result_frame, 1)
        # Show the image
        cv2.imshow('eyes', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
