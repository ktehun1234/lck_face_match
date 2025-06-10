import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# 📁 설정
IMG_SIZE = 528
dataset_dir = "crop_img"
model_path = "models/efficientnet_b6_best.h5"
class_indices_path = "models/class_indices.npy"
os.makedirs("models", exist_ok=True)

# 클래스명 추출 및 저장
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
class_indices = {name: idx for idx, name in enumerate(class_names)}
np.save(class_indices_path, class_indices)
print("🗂 클래스 인덱스 저장 완료:", class_indices)

# 🧼 이미지 로딩
X = []
y = []

for label, name in enumerate(class_names):
    folder = os.path.join(dataset_dir, name)
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(folder, fname)
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
                arr = preprocess_input(np.array(img))
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"❌ 로딩 실패: {img_path}, 오류: {e}")

X = np.array(X)
y = np.array(y)
print(f"✅ 총 이미지 수: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# ▶️ 클래스 가중치 계산
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator()

train_data = train_gen.flow(X_train, y_train, batch_size=16)
val_data = val_gen.flow(X_test, y_test, batch_size=16)

base = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False  # 일부 레이어만 학습

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)

# 학습
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# 💾 모델 저장
model.save(model_path)
print(f"💾 모델 저장 완료: {model_path}")

# 🧪 최종 평가
loss, acc = model.evaluate(val_data)
print(f"📊 테스트 정확도: {acc:.4f}")

# 📊 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("EfficientNetB6 학습 정확도", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()