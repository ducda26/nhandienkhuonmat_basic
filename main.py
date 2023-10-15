import cv2
import face_recognition

#Step 01:
imgElon = face_recognition.load_image_file("./pic/elon musk.jpg")
imaElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgCheck = face_recognition.load_image_file("./pic/elon check.jpg")
imgCheck = cv2.cvtColor(imgCheck, cv2.COLOR_RGB2BGR)

cv2.imshow("Elon", imgElon) # view thử ảnh để kiểm tra
cv2.imshow("ElonCheck", imgCheck)
cv2.waitKey()
cv2.destroyAllWindows()