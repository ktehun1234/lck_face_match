import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import requests
import urllib.parse

# 페이지 설정
st.set_page_config(page_title="LCK 얼굴 & 챔피언 마스터리 앱", layout="centered")

# 모델 및 경로
MODEL_PATH = "models/efficientnet_b6_best.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"
FACE_REP_DIR = "crop_img"

# Riot API 설정
# API_KEY = "RGAPI-20b86928-02fe-4206-9dd3-4db28595501b"
# REGION = "kr"  # Riot API 호출 시에는 'kr' 사용
# SUMMONER_BY_NAME = "https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{name}"
# MASTERY_BY_PUUID = "https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top"

# 모델 로드
@st.cache_resource
def load_face_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_map():
    data = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
    return {v: k for k, v in data.items()}

face_model = load_face_model()
class_map = load_class_map()

# 얼굴 검출 함수
def detect_and_crop_face(img: Image.Image):
    arr = np.array(img)
    if arr.shape[0] < 300:
        arr = cv2.resize(arr, (arr.shape[1]*2, arr.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    crop = arr[y:y+h, x:x+w]
    return Image.fromarray(crop).resize((528, 528), Image.LANCZOS)

# UI 입력
st.title("나와 닮은 lck선수 알아보기")
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("얼굴 이미지 업로드", type=["jpg","jpeg","png"])
with col2:
    summoner_name = st.text_input("소환사 닉네임 입력")

# 얼굴 예측
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='원본 이미지', use_column_width=True)
    face = detect_and_crop_face(img)
    if face is None:
        st.error("얼굴을 인식하지 못했습니다. 정면 사진을 업로드해주세요.")
    else:
        st.image(face, caption='크롭된 얼굴', width=224)
        inp = preprocess_input(np.array(face))
        inp = np.expand_dims(inp, 0)
        preds = face_model.predict(inp)[0]
        idx = int(np.argmax(preds))
        name = class_map[idx]
        conf = preds[idx] * 100
        st.markdown(f"## 닮은 선수: **{name}** ({conf:.1f}% 확신)")
        st.markdown("### 유사 TOP 5")
        for rank, i in enumerate(np.argsort(preds)[::-1][:5], 1):
            st.markdown(f"{rank}. **{class_map[i]}**: {preds[i]*100:.1f}%")

# 챔피언 마스터리 예측
# if summoner_name:
#     headers = {"X-Riot-Token": API_KEY}
#     encoded = urllib.parse.quote(summoner_name)
#     url = SUMMONER_BY_NAME.format(region=REGION, name=encoded)
#     resp = requests.get(url, headers=headers)
#     try:
#         resp.raise_for_status()
#     except Exception as e:
#         st.error(f"소환사 조회 실패: {e}")
#     else:
#         puuid = resp.json().get('puuid')
#         murl = MASTERY_BY_PUUID.format(region=REGION, puuid=puuid)
#         mresp = requests.get(murl, headers=headers)
#         try:
#             mresp.raise_for_status()
#         except Exception as e:
#             st.error(f"챔피언 마스터리 조회 실패: {e}")
#         else:
#             masteries = mresp.json()
#             st.markdown("---")
#             st.subheader("🏆 챔피언 마스터리 Top 5")
#             versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
#             curr = versions[0]
#             champ_resp = requests.get(f"http://ddragon.leagueoflegends.com/cdn/{curr}/data/en_US/champion.json").json()
#             key2name = {int(v['key']): v['id'] for v in champ_resp['data'].values()}
#             for m in masteries[:5]:
#                 cid = m.get('championId')
#                 cname = key2name.get(cid, str(cid))
#                 lvl = m.get('championLevel')
#                 pts = m.get('championPoints')
#                 st.write(f"- {cname}: 레벨 {lvl}, {pts:,} 포인트")
