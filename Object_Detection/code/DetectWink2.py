import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
glass_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


# ROI:picture
# /Users/dijin/PycharmProjects/ComputerVision/venv/lib/python3.6/site-packages/cv2/data
# /Users/dijin/Desktop/CV/project/2/img

def detectWink(frame, location, ROI, cascade):
    # sharpness = cv2.Laplacian(ROI, cv2.CV_64F).var()
    # # print("Face sharpness: ",sharpness)
    # if sharpness < 20:
    #     kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    #     ROI = cv2.filter2D(ROI, -1, kernel)
    eyes = cascade.detectMultiScale( ROI, 1.1, 30, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
    if len(eyes)==0:
        eyes = glass_cascade.detectMultiScale(ROI, 1.05, 1, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))
    if len(eyes)>=2:
        eyes = glass_cascade.detectMultiScale(ROI, 1.1, 4, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))
    # if len(eyes)>=2:
    #     n=len(eyes)
    #     for i in range(n - 1):
    #         for j in range(i+1,n - 1):

    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return len(eyes) == 1  # number of eyes is one


def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    #gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.1  # range is from 1 to ..
    minNeighbors = 3  # range is from 0 to ..
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30, 30)  # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + int(h/1.5), x:x + w]  ###############################
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount


def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while (showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                         + 'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + 'haarcascade_eye.xml')

    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, eye_cascade)
