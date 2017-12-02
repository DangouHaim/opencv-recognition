import numpy
import scipy
import cv2
import skvideo.io
import scipy.misc

eye_casc = cv2.CascadeClassifier("eye.xml")
face_casc = cv2.CascadeClassifier("fface_default.xml")

cap = skvideo.io.vread("data.mp4")



print("start")
for img in cap:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_casc.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		print("detected face")
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

		sub_img = gray[x:x+w, y:y+h]
		sub_imgrgb = img[x:x+w, y:y+h]
		
		eyes = eye_casc.detectMultiScale(sub_img)
		for (xe, ye, we, he) in eyes:
			cv2.rectangle(sub_imgrgb, (xe, ye), (xe+we, ye+he), (0, 255, 0), 2)
			print("detected eye")

	scipy.misc.imsave('outfile.jpg', img)
	cv2.imshow("img", img)

print("exit")

cv2.destroyAllWindows()