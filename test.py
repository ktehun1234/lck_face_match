import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from collections import Counter

# 1) 경로 설정
MODEL_PATH = "models/efficientnet_b6_best.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"
DATA_DIR = "crop_img"   # crop_faces.py로 만들어진 얼굴 폴더

# 2) 모델ㆍ클래스 로드
model = load_model(MODEL_PATH)
class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
idx_to_class = {v:k for k,v in class_indices.items()}

# 3) 예측 분포 집계용
pred_counter = Counter()

# 4) 모든 이미지 순회하며 예측
for cls, idx in class_indices.items():
    folder = os.path.join(DATA_DIR, cls)
    for fn in os.listdir(folder):
        if not fn.lower().endswith(('.jpg','jpeg','png')):
            continue

        path = os.path.join(folder, fn)
        # 이미지 로드 및 전처리
        img = Image.open(path).resize((528, 528)).convert("RGB")
        arr = np.array(img)
        arr = preprocess_input(arr)                # EfficientNet 전처리
        arr = np.expand_dims(arr, 0)

        # 예측
        preds = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_name = idx_to_class[pred_idx]
        confidence = preds[pred_idx]

        pred_counter[pred_name] += 1
        print(f"{cls:15s} | {fn:20s} → {pred_name:10s} ({confidence*100:5.1f}%)")

print("\n=== 전체 예측 분포 ===")
for name, count in pred_counter.most_common():
    print(f"{name:10s}: {count}")

# 5) Dense 레이어 bias 읽어서 출력
dense = model.layers[-1]
# 모델 구조에 따라 final layer가 다를 수 있으니, 확인 후 조정하세요
weights, biases = dense.get_weights()
# bias가 큰 순서대로 top5
top5 = np.argsort(biases)[::-1][:5]
print("\n=== Bias 가 큰 상위 5개 클래스 ===")
for i in top5:
    print(f"{idx_to_class[i]:10s}: bias={biases[i]:.3f}")