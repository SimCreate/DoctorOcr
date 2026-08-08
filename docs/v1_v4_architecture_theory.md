# DoctorOcr v1~v4 아키텍처 이론 (교차검증 정리)

> 작성: 2026-08-08
> 방법: 우리 실측(코드·결과) + ChatGPT(웹검색) + 로컬 V4 Flash(독립 팩트체크) 이중 교차검증
> 목적: v1→v4 인코더/디코더/손실 각 구성 요소의 이론적 배경과 근거 논문 정리

---

## 1. 전체 구조

입력 처방전 이미지 X → CNN 특징 추출 → 시퀀스 모델링 → transcription 순서다.

```
X ──▶ CNN(시각적 특징) ──▶ BiLSTM(좌우 문맥) ──▶ Attention Decoder(문자열 생성)
                                                └─▶ CTC Head(정렬-자유 정렬)
손실: L = λ·L_ctc + (1-λ)·β·L_ce
```

각 구성 요소는 서로 다른 종류의 **inductive bias**를 결합한 것으로 볼 수 있다.

| 구성 | inductive bias | 해결하는 문제 |
|---|---|---|
| CNN | 시각적 국소성 | 획/에지/문자패턴 추출 |
| SEBlock | 채널 중요도 재가중 | 어떤 채널을 믿을지 |
| ResNet | 잔차 학습 | 깊은 네트워크 최적화 |
| BiLSTM | 좌우 문맥 | 앞뒤 글자 의존성 |
| Attention | soft alignment | 출력에 필요한 이미지 위치 선택 |
| CTC | monotonic 정렬 | 위치 라벨 없이 정렬-자유 학습 |
| Beam search | 시퀀스 전체 탐색 | 글자별 greedy의 한계 |

## 2. v1→v4 아키텍처 계보 (코드 기반)

인코더 구성

| 단계 | CNN 인코더 | 시퀀스 인코더 | 데이터 (train) |
|---|---|---|---|
| v1 | 7 Conv Block (→512ch) | BiLSTM 2층, hidden 256 | 89장 |
| v2 | SEBlock 5블록 (→512ch) | BiLSTM 3층, hidden 384 | 5,578장 |
| v3 | v3_1: SEBlock → v3_2: resnet18 | BiLSTM 3층, hidden 384 | 13,386장 |
| v4 | resnet18 (ImageNet pretrained, layer3) | BiLSTM 3층, hidden 384 | 13,386장 (라벨 정제) |

디코더 구성

| 단계 | 디코더 | 손실 | exact (클린 val) |
|---|---|---|---|
| v1 | AttentionDecoder 4-head (LSTM1) | CE | 0% |
| v2 | AttentionDecoder 8-head (LSTM2) + Beam | CE | ~20% |
| v3 | AttentionDecoder 8-head + Beam | CE | 35.8~37.9% |
| v4 | AttentionDecoder 8-head + CTCHead (하이브리드) | λ·L_ctc + (1-λ)·β·L_ce | attn 62.3% / CTC 48.2% |

> 실제 이력상 CTC는 v1/v2의 attention 계열과 별개로 v2_2(레거시, CTC 전용)에서 먼저 검증됐고, v4에서 attention과 공동 손실로 통합됐다. v1부터 CTC였던 구조가 아니다.

---

## 3. 구성 요소별 이론적 배경

### 3.1 CNN 인코더 — 7Conv → SEBlock → resnet18

**왜 CNN인가**: 처방전 이미지는 2차원 신호다. 획의 방향·곡률·교차점·굵기 같은 local visual pattern이 중요하다. CNN은 local receptive field로 이런 특징을 계층적으로 추출한다 — 초기 레이어는 edge/stroke, 깊어질수록 문자 부분·문자 형태로 receptive field가 커진다. 원조 CRNN(Shi et al. 2015)은 문자 단위 segmentation 없이 가변길이 시퀀스를 처리하는 것이 핵심 장점이었다.

**7Conv의 의미**: 특별한 이론적 숫자가 아니라 충분한 convolution depth를 확보하는 설계 선택이다. 손글씨에서 `rn`과 `m`처럼 작은 local patch만 보면 구분이 어려운 경우, 깊은 CNN이 여러 패턴을 조합해 더 넓은 context를 얻는다. 한계는 long-range dependency 부족, 명시적 sequence ordering 모델 부재, 과한 pooling에 의한 세부 획 정보 손실이다.

**SEBlock (Hu et al. 2018, Squeeze-and-Excitation Networks)**: CNN이 spatial 정보와 channel 정보를 함께 추출하지만 채널 간 중요도를 명시적으로 모델링하지 못한다는 문제에서 출발했다. 세 단계로 작동한다.
- Squeeze: 각 채널을 spatial global average pooling으로 스칼라 하나에 압축
- Excitation: 작은 MLP + sigmoid로 채널별 중요도 s ∈ [0,1] 학습
- Recalibration: 채널 출력에 s를 곱해 재조정

채널 A가 획을 잘 잡고 B가 배경 노이즈를 잡는다면, SE가 A에 높은 가중치를 준다. **주의**: spatial attention이 아니라 channel attention이다. "이미지의 x=130 위치가 중요하다"를 선택하는 게 아니라 "이 feature channel이 중요하다"를 선택한다 (교차검증: 로컬 LLM이 "SEBlock은 인코더 단계가 아니라 CNN 내부 모듈"로 정정). SE 자체는 sequence ordering이나 문자 정렬을 해결하지 않는다 — 이후 BiLSTM/Attention/CTC 담당.

**ResNet18 (He et al. 2015)**: 핵심은 잔차 학습 y = F(x) + x. 네트워크가 H(x)를 직접 학습하는 대신 잔차 F(x) = H(x) − x를 학습해 깊은 네트워크의 최적화를 안정화한다. v3에서 256x64 무조건 리사이즈(4:1 왜곡)를 256x128 비율유지 패딩으로 바꾸고, SEBlock CNN을 ImageNet pretrained resnet18로 교체했다 — 사전학습 backbone이 저수준 특징(획/에지) 전이에 효과적이라는 교차검증(2026-08-07) 합의에 따른 것. 한계: (a) stride conv 다운샘플링에 의한 해상도 손실 — `ll` 같은 작은 차이를 지울 수 있음 (b) 일반 분류용 backbone이라 width 방향 해상도를 과도하게 줄이는 것을 피해야 함 (c) ImageNet(자연 이미지)과 흑백/필기체의 도메인 미스매치 — fine-tuning 필요.

### 3.2 BiLSTM — 시퀀스 인코더

CNN 특징맵을 width 방향으로 펼치면 x₁,…,x_T 시퀀스가 된다. 일본어/영어 필기체에서 현재 글자를 읽을 때 앞뒤 문맥이 모두 필요하다 — 어떤 획이 a/o/e 중 무엇인지 local image만으로 애매할 수 있기 때문이다.

- **LSTM**: 일반 RNN은 긴 시퀀스에서 gradient vanishing이 생기는데, LSTM은 cell state에 정보를 선택적으로 보존(i/f/o 게이트)해 완화한다.
- **BiLSTM**: 순방향 LSTM과 역방향 LSTM을 동시에 계산하고 은닉 상태를 결합([h→; h←])해 왼쪽+오른쪽 문맥을 모두 사용한다. 필기체에서 현재 문자가 애매할 때 오른쪽 글자를 함께 보면 판단이 쉬워진다.

단점은 순차 계산이라 Transformer처럼 병렬화가 어렵다는 것. 이것이 이후 SVTR이 sequence model 자체를 제거하고 visual token mixing만으로 인식하는 방향으로 발전한 배경과 연결된다.

### 3.3 Seq2Seq + Attention 디코더

CTC가 "이미지 feature를 문자로 정렬"에 가깝다면, Seq2Seq는 "지금까지 생성한 문자와 이미지의 어느 부분을 봐야 다음 문자가 나오는가"를 학습한다.

- **Encoder-Decoder**: P(Y|X) = ∏_u P(y_u | y_<u, X). 현재 문자의 확률이 이전에 생성한 문자에 의존한다. 이것이 CTC와의 가장 중요한 차이다 (문자 간 조건부 의존성).
- **Bahdanau Attention (2014)**: encoder 전체를 하나의 고정 벡터로 압축하면 정보 병목이 생긴다는 문제에서 출발. decay가 매 step마다 필요한 source 위치를 soft하게 선택한다. additive alignment score.
- **Luong Attention (2015)**: dot-product 기반의 단순화된 attention (global/local).
- **Multi-Head Attention (Vaswani et al. 2017)**: Q=XWq, K=XWk, V=XWv로 투영 후 Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V. 여러 head가 서로 다른 project space에서 local stroke 관계, 문자 간 관계, 장거리 dependency를 동시에 볼 수 있다. 우리 프로젝트는 v1(4-head) → v2 이상(8-head)로 확장했다.
- **LAS (Chan et al. 2015, Listen Attend and Spell)**: Listener=인코더, Attend=어텐션, Speller=디코더 구조로 음성을 문자 시퀀스로 직접 변환. attention 기반 디코더가 출력 history를 사용해 문자 간 의존성을 명시적으로 모델링한다는 점이 핵심. 우리 CNN→BiLSTM→Attention 구조는 LAS 철학을 이미지 시퀀스 인식으로 옮긴 것.

### 3.4 CTC — Connectionist Temporal Classification (Graves et al. 2006)

음성인식의 "몇천 개 acoustic frame → HELLO"처럼 입력 길이(T)와 출력 길이(U)가 다르고, 어느 위치가 어느 문자에 대응하는지 라벨이 없는 문제에서 출발했다. OCR도 동일하다 — T개 visual timestep과 U개 문자의 정확한 정렬이 없다.

- **Blank**: 반복 문자를 표현하기 위한 특수 심볼. `CAT`→`CC--AA-TT`처럼, `BOOK`의 OO는 `O-O`로 표현해야 한다. CTC는 중복 제거 + blank 제거로 정렬을 collapse한다.
- **Alignment marginalization**: 하나의 정렬을 정답으로 강제하지 않고 가능한 모든 경로 π를 합산해 P(Y|X)=Σ_{π∈B⁻¹(Y)}P(π|X)를 계산한다. 이 합은 동적 계획법으로 효율적으로 계산된다. → **문자별 위치 라벨 없이 sequence-level 라벨만으로 학습 가능**.
- **조건부 독립 가정**: P(π|X)=∏_t P(π_t|X). 현재 timestep의 문자 확률이 이전 출력 문자에 직접 의존하지 않는다. 이것이 장점(정렬 불필요, 학습 안정, 추론 빠름)이자 한계(문자 간 언어 의존성 미모델링 — amoxicillin의 앞 글자가 뒤 글자에 주는 제약을 활용 못함)다.
- CTC의 monotonic(좌→우) 정렬 bias는 손글씨·음성 인식에 잘 맞는다.

### 3.5 Attention vs CTC

| | CTC | Attention Seq2Seq |
|---|---|---|
| 정렬 | implicit, monotonic 강한 bias | learned soft alignment |
| 출력 의존성 | 약함 (조건부 독립) | 강함 (이전 문자 직접 활용) |
| 학습 안정성 | 비교적 안정 | 초기 alignment 학습이 어려울 수 있음 |
| 디코딩 | 빠름 | 상대적으로 복잡 (탐색 필요) |
| 긴 문맥 | 제한적 | 강함 |

한 문장으로: **CTC는 정렬을 단순·monotonic하게 만들면서 모든 alignment를 marginalize하고, Attention은 출력할 때마다 필요한 입력 위치를 직접 선택한다.**

### 3.6 Hybrid CTC/Attention — 공동 손실 (v4 핵심)

v4는 공유 인코더(resnet18+BiLSTM)에 두 헤드를 병렬로 붙인다.

```
Shared Encoder ──┬── CTC loss
                 └── Attention loss
L = λ·L_ctc + (1-λ)·β·L_ce
```

**이론적 근거 (Kim, Hori, Watanabe 2016; Watanabe et al. 2017)**: 넓은 의미에서 shared encoder + 서로 다른 supervision head를 쓰는 **multi-task learning**이다. 두 task가 완전히 다른 것은 아니지만 같은 인코더 표현에 서로 보완적인 학습 신호를 제공한다. CTC는 attention의 초기 alignment 문제를 보완하고 수렴·강건성을 개선한다.

- **λ = CTC 비중**: λ=1이면 CTC만, λ=0이면 Attention만. λ=0.3이면 CTC 30% / Attention 70%. 우리 프로젝트는 0.5→0.3으로 낮춰 attention 부활을 장려했다 (교차검증·학습 실측: attention 부활에 λ=0.3이 유효).
- **β = CE 스케일 (우리 코드 기준)**: hybrid_loss 주석에 명시된 대로 "CE는 만점(0)으로 수렴하기 쉬워 그래디언트가 죽음 → 독립 스케일로 보존"하기 위한 배율. β=3.0. **주의**: 로컬 LLM은 "β는 일반적으로 joint CTC/attention decoding의 language model 가중치"로 지적했지만, 이는 우리 코드를 모르는 상태의 일반론이다. 우리 프로젝트에서 β는 loss 식 안의 CE 배율이며(코드 검증), Hori et al. 2017의 joint decoding에서 쓰는 β는 디코딩 단계의 별개 개념이다. 문서·코드에서 혼동하지 말 것.

CPU상 CTC는 "어디에 어떤 문자가 있는가"에 대한 정렬 regularizer, Attention은 "지금 무엇을 출력할지 + 어떤 시각적 위치를 볼지"의 강한 자동회귀 시퀀스 모델링으로 이해하는 것이 정확하다.

### 3.7 Beam Search

Attention 디코더는 매 timestep P(y_t | y_<t, X)를 계산한다. Greedy는 가장 확률 높은 한 글자만 고르는데, 이는 전체 시퀀스 확률을 최적화하지 못한다. Beam search는 매 timestep마다 상위 B개(beam width) 가설만 유지하며 전체 조합(|V|^T은 불가능)을 근사 탐색한다. 점수는 보통 score(Y)=Σ_t log P(y_t | y_<t, X). 우리 v2부터 beam_width=5 채택. Hori et al. 2017은 hybrid에서 CTC score와 attention score를 디코딩 단계에서 결합하는 joint decoding을 제안했다 (우리는 현재 beam5 attention 단독 — CTC 결합 디코딩은 미도입).

### 3.8 차세대 방향 — TrOCR / SVTR (비교축)

- **TrOCR (Li et al. 2021/2022)**: CNN→RNN→문자 디코더 계열에서 벗어나 vision(Image Transformer)과 text(Text Transformer)를 모두 Transformer로 처리하고 pretrained 모델을 활용한다. 우리 v4의 다음 세대 방향으로 자연스럽다.
- **SVTR (Du et al. 2022)**: sequence model을 제거하고 visual token의 local/global mixing만으로 텍스트 인식. "BiLSTM이 최신 OCR에서 반드시 필요한가"라는 근본 질문을 던진다.
- 비교축: CRNN+CTC / CRNN+Attention / CRNN+CTC+Attention / SVTR / TrOCR 5-way가 후속 실험의 자연스러운 구성.

---

## 4. 우리 프로젝트에 적용된 교훈 (교차검증 반영)

- **시퀀스 시프트 버그**: attention(순차 모델)은 학습 시 입력/정답 시프트 일치가 생명. 정답을 targets[:,1:]로 시프트하기 전엔 모델이 <SOS>→<SOS> 자기복사만 학습해 beam 0%였음.
- **의료 도메인 특수성**: 비정상 문자 간격, 필기체 연결, `rn↔m`, `cl↔d` 같은 segmentation ambiguity, amoxicillin 같은 의료 vocabulary가 중요. Attention 디코더는 언어 prior가 유리할 수 있으나 동시에 이미지에 없는 글자를 "그럴듯하게" 만들어내는 **hallucination 위험이 있다** — CTC branch를 함께 두는 것이 단순 성능 trick 이상의 의미를 가진다.
- **숫자 오류가 치명적**: 500mg / 5mg / 0.5mg는 작은 시각 오류가 큰 의미적 오류가 된다.
- **저빈도 붕괴** (attn 0.9% / CTC 2.8%)는 디코더 무관의 데이터/라벨 문제로 확정 — 다음 단계 핵심 타깃.

---

## 5. 교차검증 판정 정리 (맞고 틀린 것)

### ChatGPT vs 로컬 V4 Flash — 두 LLM 모두 "정확" 합의한 것
- CTC의 blank·정렬 marginalization·조건부 독립 이론
- Hybrid를 multi-task learning 관점으로 보는 것
- SE가 channel attention이지 spatial attention이 아닌 것
- Beam search의 근사 탐색 원리
- Bahdanau(additive)/Luong(dot) attention 구분
- 참고문헌 존재 (CRNN·TrOCR·SVTR·ResNet·SE·LAS·CTC·Hybrid 모두 실존)

### 로컬 V4 Flash가 ChatGPT를 수정한 지점 (반영 완료)
1. **SEBlock 위치**: "인코더 단계"가 아니라 CNN 내부의 채널 어텐션 모듈 → 본문 3.1에 반영
2. **ResNet "pooling으로 문자 정보 손실"**: 정확히는 stride conv/downsampling에 의한 해상도 손실 → 3.1에 반영
3. **β 의미**: 로컬 LLM은 "일반적으로 joint decoding의 LM 가중치"라고 했으나, 우리 코드에서 β=3.0은 loss 식 안의 CE 배율 → 우리 코드 기준으로 3.6에 명시
4. **진화 순서**: 채택하지 않음 — 로컬 LLM도 "v1이 7Conv+BiLSTM+CTC였다"고 가정했는데, **실제 우리 v1은 attention 디코더 기반(CE)이고 CTC는 v2_2에서 먼저 검증 후 v4에서 통합**됨. 두 LLM 모두 우리 이력을 모른 채 일반화했으므로 이 문서는 코드 기반 계보(2장)를 표준으로 삼음.

### 우리 프로젝트 이력과의 정합성 (문서 작성 시 정정)
- ChatGPT의 진화 해석 "visual locality→channel selection→deep representation→bidirectional context→soft alignment→language dependency→alignment regularization" 은 우리 실제 이력과 정확히는 다름. 실제 순서: 7Conv(v1) → bidirectional context+soft alignment(v1 attention) → channel selection(SEBlock, v2) → deep representation(resnet18, v3) → alignment regularization+autoregressive 결합(CTC+attention 하이브리드, v4).

---

## 6. 참고문헌

### CRNN / OCR
- Shi, Bai, Yao (2015), *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition* — arXiv:1507.05717
- Li et al. (2021/2022), *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*
- Du et al. (2022), *SVTR: Scene Text Recognition with a Single Visual Model* — arXiv:2205.00159

### CNN
- He et al. (2015), *Deep Residual Learning for Image Recognition* — arXiv:1512.03385
- Hu, Shen, Sun (2018), *Squeeze-and-Excitation Networks* — arXiv:1709.01507

### Attention / Seq2Seq
- Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* — arXiv:1409.0473
- Luong, Pham, Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation*
- Chan et al. (2015), *Listen, Attend and Spell* — arXiv:1508.01211
- Vaswani et al. (2017), *Attention Is All You Need* — arXiv:1706.03762

### CTC / Hybrid
- Graves et al. (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*
- Kim, Hori, Watanabe (2016), *Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning*
- Watanabe et al. (2017), *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition*
- Hori, Watanabe, Hershey (2017), *Joint CTC/attention decoding for end-to-end speech recognition*
