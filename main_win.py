import numpy
import cv2


eye_casc = cv2.CascadeClassifier("eye.xml")
face_casc = cv2.CascadeClassifier("fface_default.xml")

cap = cv2.VideoCapture(0)

if cap.isOpened():
	print("start")
	while cap.isOpened():
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_casc.detectMultyScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

			sub_img = gray[x:x+w, y:y+h]
			eyes = eye_casc.detectMultyScale(sub_img)
			for (xe, ye, we, he) in eyes:
				cv2.rectangle(sub_img, (xe, ye), (xe+we, ye+he), (0, 255, 0), 2)

		cv2.imshow("img", img)

print("exit")

cap.release()
cv2.destroyAllWindows()