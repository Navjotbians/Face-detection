import argparse
import cv2
import dlib

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-w', '--weights', required=True)
args = ap.parse_args()

image = cv2.imread(args.image)

if image is None:
    print("Can not read the image")
    exit()

hog_face_detector = dlib.get_frontal_face_detector()

faces_hog = hog_face_detector(image, 1)

for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y+50

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 1)

    img_height, img_width = image.shape[:2]
    cv2.putText(image, "HOG", (img_width - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                (0, 0, 255), 8)

    cv2.imshow("face detection with HOG", image)
    cv2.waitKey()







