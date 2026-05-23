import cv2
import numpy as np
import os
import handTracking as ht

brushThickness = 12
eraserThickness = 100
folderPath = "header"
drawColor = (255, 0, 255)

myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (1280, 125))
    overlayList.append(image)

header = overlayList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Virtual Board", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

detector = ht.handDetector(detectionCon=0.9, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness

            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    img[0:125, 0:1280] = header
    cv2.imshow("Virtual Board", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

