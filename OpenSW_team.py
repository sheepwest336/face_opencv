import cv2  
def detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )  # 얼굴 탐지
    return faces

def preprocess_face(img, box, mean_values):
    x, y, w, h = box
    face = img[y:y+h, x:x+w].copy()  # 얼굴 영역 추출
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_values, swapRB=False)
    return blob

if __name__ == "__main__":
    cascade_file = 'haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_file)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    age_categories = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
                      '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
    gender_categories = ['Male', 'Female']
