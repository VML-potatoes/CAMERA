import cv2
from ultralytics import YOLO

# 객체 탐지 모델 로드 (적절한 모델 경로로 변경)
model = YOLO("paths/yolo11s.pt")

# 웹캠(또는 비디오 파일 경로)로부터 영상 스트림 시작
# 웹캠인 경우: 0, 비디오 파일인 경우 "video.mp4"와 같이 경로를 지정
cap = cv2.VideoCapture(1)
print(cap)

if not cap.isOpened():
    print("카메라 또는 영상 파일을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽어올 수 없습니다. 스트림 종료.")
        break

    # 프레임에 대해 예측 수행
    results = model.predict(frame)

    # 첫 번째 결과에서 예측된 객체들을 이미지에 표시
    annotated_frame = results[0].plot()

    # 결과 출력
    cv2.imshow("Real-time YOLO Prediction", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
