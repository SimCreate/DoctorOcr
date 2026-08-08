# Visual Asset Attributions

각 시각 자료의 원저작자/라이선스를 기록한다. 모든 파일은 원본 저장소에서
**원본 그대로** 커밋됨 (수정·결합 없음).

| 파일 | 제목 | 원저작자 | 라이선스 | 출처 |
|---|---|---|---|---|
| `cnn_convolution.gif` | Convolution arithmetic — full padding, no strides (커널 슬라이딩 애니메이션, 49프레임) | Vincent Dumoulin, Francesco Visin | MIT (Expat) | https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/full_padding_no_strides.gif |
| `lstm_cell.svg` | NN LSTM-Cell v2 | Leouscin | CC0 1.0 (퍼블릭 도메인) | https://commons.wikimedia.org/wiki/File:NN_LSTM-Cell_v2.svg |
| `attention.gif` | Attention-animated | Numiri | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Attention-animated.gif |
| `beam_search.gif` | Beam search | BogdanShevchenko | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Beam_search.gif |

## 라이선스 전문

- **MIT** (cnn_convolution.gif): 복제·수정·배포·판매 가능. 저작권 고지 및 라이선스 문구 보존 필요.
  - LICENSE: https://github.com/vdumoulin/conv_arithmetic/blob/master/LICENSE (The MIT License, Copyright (c) 2016)
- **CC0 1.0** (lstm_cell.svg): 사실상 퍼블릭 도메인. 복제·수정·상업적 이용·재배포 가능. attribution 법적 의무 없음.
- **CC BY-SA 4.0** (attention.gif, beam_search.gif): 복제·수정·재배포 가능. 저작자 표시 + 라이선스 링크 필요.
  수정물 배포 시 동일 라이선스(SA) 조건. **원본 그대로 사용 중이므로 재배포 적법.**

## 검증 기록 (2026-08-08)

교차검증 중 검증된 사실을 기록한다.

1. **ChatGPT(웹검색)가 제안한 "Convolution arithmetic - No padding strides"** 파일명은
   실제 Wikimedia에는 **없음** (404). Dumoulin github 원본의 실제 파일명은
   `no_padding_no_strides.gif`(4프레임, 정적 도식) / `full_padding_no_strides.gif`(49프레임, 커널 슬라이딩 애니메이션).
   → **커널이 이동하는 동작을 보여주는 것은 `full_padding_no_strides.gif`** 로 교체해 사용.
2. **프레임 검증**: 각 GIF를 프레임 단위로 확인.
   - `cnn_convolution.gif`: 49프레임, frame 0 커널=좌상단 → frame 48 커널=우하단 (픽셀 이동 확인)
   - `attention.gif`: 21프레임
   - `beam_search.gif`: 4프레임, frame 0 전체 트리 → frame 3 가지치기(빨간 활성 3경로 + 회색 제거 노드)
3. 모든 파일은 커밋 전 원본 URL(HTTP 200)과 라이선스 페이지에서 실재 확인됨.

## md에서의 사용 위치

- `docs/v1_v4_architecture_theory.md` — 3장 구성요소별 이론 (CNN → LSTM → Attention → CTC → Beam search) 각 절에 삽입.
