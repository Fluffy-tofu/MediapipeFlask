import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

        self.landmark_positions = []

    def findPos(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True, draw_path=False, landmark_draw_path=23, num_points=30):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if id == landmark_draw_path:
                    self.landmark_positions.append([cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

        if draw_path and len(self.landmark_positions) >= 2:
            for i in range(len(self.landmark_positions) - 1):
                x1, y1 = self.landmark_positions[i]
                x2, y2 = self.landmark_positions[i + 1]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return lmList, img

def main():
    cap = cv2.VideoCapture('PoseVideos/Video5.MOV')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        img_copy = img.copy()  # Create a copy of the original image

        img_copy = detector.findPos(img_copy)
        lmList, img_copy = detector.getPosition(img_copy, draw_path=True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img_copy, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow('Image', img_copy)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()