from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("동영상.mp4")

# 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("결과.mp4", fourcc, fps, (w, h))

def mosaic(img, x1, y1, x2, y2, block_size=20):
    """해당 영역을 블록 처리"""
    roi = img[y1:y2, x1:x2]
    # 작게 줄였다가 다시 키우면 블록 효과
    small = cv2.resize(roi, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    mosaic_roi = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic_roi
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        frame = mosaic(frame, x1, y1, x2, y2, block_size=15)  # 숫자 낮을수록 더 뭉개짐

    out.write(frame)
    cv2.imshow("결과", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # q 누르면 종료
        break

cap.release()
out.release()
cv2.destroyAllWindows()