import cv2
import numpy as np
import dlib
from math import hypot

def midPoint(a, b):
    return int((a.x + b.x)/2), int((a.y + b.y)/2)

def blinkingRatio(eyePoints, landmarks):
    #mengambiil titik-titik bagian mata dari landmarks yang telah tersedia
    leftPoint = (landmarks.part(eyePoints[0]).x, landmarks.part(eyePoints[0]).y)
    RightPoint = (landmarks.part(eyePoints[3]).x, landmarks.part(eyePoints[3]).y)
    topPoint = (midPoint(landmarks.part(eyePoints[1]), landmarks.part(eyePoints[2])))
    bottomPoint = (midPoint(landmarks.part(eyePoints[5]), landmarks.part(eyePoints[4])))

    #menggambar garis horizontal dan vertikal mata
    horLine = cv2.line(frame, leftPoint, RightPoint, (0, 255, 0), 2)
    verLine = cv2.line(frame, topPoint, bottomPoint, (0, 255, 0), 2)

    #menghitung rasio dari garis horizontal mata dengan vertikal mata
    horLineLength = hypot((leftPoint[0] - RightPoint [0]), (leftPoint[1] - RightPoint[1]))
    verLineLength = hypot((topPoint[0] - bottomPoint[0]), (topPoint[1] - bottomPoint[1]))
    ratio = horLineLength/verLineLength

    return ratio

def gazeRatio(eyePoints, landmarks):
    #Area mata kiri berdasarkan landmarks
        EyeRegion = np.array([(landmarks.part(eyePoints[0]).x, landmarks.part(eyePoints[0]).y),
                              (landmarks.part(eyePoints[1]).x, landmarks.part(eyePoints[1]).y),
                              (landmarks.part(eyePoints[2]).x, landmarks.part(eyePoints[2]).y),
                              (landmarks.part(eyePoints[3]).x, landmarks.part(eyePoints[3]).y),
                              (landmarks.part(eyePoints[4]).x, landmarks.part(eyePoints[4]).y),
                              (landmarks.part(eyePoints[5]).x, landmarks.part(eyePoints[5]).y)], np.int32)
        
        #garis polygon sekitar mata
        #cv2.polylines(frame, [EyeRegion], True, (255, 0, 0), 2)
        
        #membuat frame yang hanya diisi oleh daerah mata
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [EyeRegion], True, 255, 2)
        cv2.fillPoly(mask, [EyeRegion], 255)
        Eye = cv2.bitwise_and(gray, gray, mask = mask)
        
        #mencari titik ekstrim dari bagian mata
        min_x = np.min(EyeRegion[:, 0])
        max_x = np.max(EyeRegion[:, 0])
        min_y = np.min(EyeRegion[:, 1])
        max_y = np.max(EyeRegion[:, 1])

        EyeEkstrim = Eye[min_y : max_y, min_x : max_x]
        _, thresholdEye = cv2.threshold(EyeEkstrim, 70, 255, cv2.THRESH_BINARY)
        
        #mencari lebar dan tinggi dari threshold agar bisa dibagi 2 sama besar
        EyeHeight, EyeWidth = thresholdEye.shape 

        #sisi bagian kiri threshold dihitung bagian putihnya        
        leftSideThreshold = thresholdEye[0 : EyeHeight, 0 : int(EyeWidth/2)]
        leftSideWhite = cv2.countNonZero(leftSideThreshold)
        
        #sisi bagian kanan threshold dihitung bagian putihnya
        rightSideThreshold = thresholdEye[0 : EyeHeight, int(EyeWidth/2) : EyeWidth]
        rightSideWhite = cv2.countNonZero(rightSideThreshold)
     
        #rasio warna putih bagian kiri dengan kanan
        if rightSideWhite == 0:
            rightSideWhite = 1
        elif leftSideWhite == 0:
            leftSideWhite = 1
        whiteRatio = leftSideWhite/rightSideWhite

        return whiteRatio

webcam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("/home/bakar/Documents/GazeDetection/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        ##BLINKING DETECTION##
        
        #menghitung ratio kedipan dari mata kanan dan kiri
        rightEyeRatio = blinkingRatio([36, 37, 38, 39, 40, 41], landmarks)
        leftEyeRatio = blinkingRatio([42, 43, 44 ,45, 46, 47], landmarks)
        
        #jika rasio lebih dari 6 maka akan terdeteksi berkedip
        if rightEyeRatio > 6 or leftEyeRatio > 6:
            cv2.putText(frame, "Blinking", (10,120), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

        
        ##GAZE DETECTION##
        
        #menghitung white rasio mata kiri dan kanan
        leftGazeRatio = gazeRatio([36, 37, 38, 39, 40, 41], landmarks)
        rightGazeRatio = gazeRatio([42, 43, 44, 45, 46, 47], landmarks)
        gazeRatioValue = (leftGazeRatio + rightGazeRatio)/2

        #cek arah mata dengan hasil rasio warna putih
        if gazeRatioValue <= 1:
            cv2.putText(frame, "Look Right", (10,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
        elif 1 < gazeRatioValue < 3:
            cv2.putText(frame, "Look Center", (10,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
        elif gazeRatioValue >= 3:
            cv2.putText(frame, "Look Left", (10,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break