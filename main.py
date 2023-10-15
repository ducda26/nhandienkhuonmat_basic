import cv2
import face_recognition

# Step 01:
imgElon = face_recognition.load_image_file("./pic/elon musk.jpg")
imaElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgCheck = face_recognition.load_image_file("./pic/elon check.jpg")
imgCheck = cv2.cvtColor(imgCheck, cv2.COLOR_RGB2BGR)

# Step 02: Xác định vị trí khuôn mặt cần nhận dạng
faceloc = face_recognition.face_locations(
    imgElon)[0]  # Vì có 1 bức ảnh nên index = 0
print(faceloc)  # (y1,x2,y2,x1)
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]),
              (faceloc[1], faceloc[2]), (255, 0, 255), 2)


faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck, (faceCheck[3], faceCheck[0]),
              (faceCheck[1], faceCheck[2]), (255, 0, 255), 2)


# So sánh hình ảnh mã hóa với các điểm trên khuôn mặt có khớp không
results = face_recognition.compare_faces([encodeElon], encodeCheck)
print(results)

# tuy nhiên khi có nhiều hình ảnh thì chúng ta cần phải biết
# khoảng cách (sai số ) giữa các bức ảnh là bao nhiêu?
faceDis = face_recognition.face_distance([encodeElon], encodeCheck)
print(results, faceDis)


# Hiển thị kết quả lên màn hình
cv2.putText(imgCheck, f"{results}{((1-round(faceDis[0],2)))*100}%",
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Elon", imgElon)  # view thử ảnh để kiểm tra
cv2.imshow("ElonCheck", imgCheck)
cv2.waitKey()
cv2.destroyAllWindows()
