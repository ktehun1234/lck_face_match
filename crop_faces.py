import os
import cv2
import numpy as np
from PIL import Image

# DNN 모델 경로
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

# DNN 네트워크 로드
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def crop_faces_from_dataset(input_dir='dataset', output_dir='crop_img'):
    total = 0
    saved = 0
    player_save_count = {}
    padding = 30  # 얼굴 주변 여백

    for player_name in os.listdir(input_dir):
        player_path = os.path.join(input_dir, player_name)
        if not os.path.isdir(player_path):
            continue

        output_player_path = os.path.join(output_dir, player_name)
        os.makedirs(output_player_path, exist_ok=True)

        player_saved = 0

        for file_name in os.listdir(player_path):
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
                continue

            img_path = os.path.join(player_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 이미지 로딩 실패: {img_path}")
                continue

            # 해상도 보정 (너무 작은 경우 업샘플링)
            if img.shape[0] < 300:
                img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

            faces = detect_face_dnn(img)
            total += 1

            if len(faces) == 0:
                print(f"❌ 얼굴 감지 실패: {img_path}")
                continue

            # 가장 큰 얼굴 선택 + 패딩 적용
            x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            face_crop = img[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_resized = face_pil.resize((528, 528), Image.LANCZOS)

            # 선수이름_번호 형식으로 저장
            idx = player_saved + 1
            save_name = f"{player_name}_{idx}.png"
            save_path = os.path.join(output_player_path, save_name)
            face_resized.save(save_path, format='PNG')

            saved += 1
            player_saved += 1

        player_save_count[player_name] = player_saved

    print(f"\n 전체 얼굴 crop 완료: {saved}/{total}장 저장됨 (저장 경로: {output_dir}/.)")
    print("\n 선수별 저장 결과:")
    for player, count in sorted(player_save_count.items()):
        print(f"  - {player}: {count}장")

if __name__ == "__main__":
    crop_faces_from_dataset(input_dir="dataset", output_dir="crop_img")