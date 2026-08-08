#!/usr/bin/env python3
"""
공용 전처리 — 이미지 로드 + 비율유지 패딩 리사이즈 (v4 resnet 백본용)
====================================================================
v3_1 문제: 128x128 원본을 256x64로 강제 리사이즈 → 4:1 왜곡 (글자 눌림).
v4 변경: 256x128 캔버스에 비율 유지 + 중앙 패딩 → 형태 보존.

원본이 다양한 크기면:
  - 높이를 H_target으로 맞추고 폭은 비율 유지 (최대 W_target, 넘으면 축소)
  - 캔버스 중앙에 배치, 좌우/상하 패딩

ImageNet preprocessing은 입력이 224x224 기반이라,
resnet18 backbone은 gray->RGB 3채널, Normalize(ImageNet mean/std)를 사용.
"""
import cv2
import numpy as np

# v4 모델의 입력 크기 (model_v4_resnet.py와 동일)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256

# ImageNet 표준 정규화
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_resize_pad(path, target_h=IMAGE_HEIGHT, target_w=IMAGE_WIDTH):
    """이미지 로드 → (target_w, target_h) 비율유지 패딩.

    - cv2.imread (None이면 빈 캔버스)
    - BGR→RGB
    - 높이 target_h 기준 스케일, 폭 비율 유지 (넘으면 target_w로 축소)
    - 중앙 배치 + 0(검정) 패딩
    반환: (H, W, 3) float 이미지 [0,1] 범위 (RGB)
    """
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.float32)

    # 높이 기준 스케일 (높이를 target_h로)
    scale = target_h / h
    new_w = int(round(w * scale))
    new_w = min(new_w, target_w)  # 폭 초과 방지
    if new_w < 1:
        new_w = 1

    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 중앙 배치 + 패딩
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    x0 = (target_w - new_w) // 2
    canvas[:, x0:x0 + new_w] = resized / 255.0

    return canvas  # (H, W, 3) float [0,1] RGB


def preprocess_tensor(img_rgb_01):
    """(H,W,3) float [0,1] RGB → (3,H,W) ImageNet-normalized tensor."""
    # HWC -> CHW
    x = img_rgb_01.transpose(2, 0, 1)
    # ImageNet normalize
    for c in range(3):
        x[c] = (x[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return x
