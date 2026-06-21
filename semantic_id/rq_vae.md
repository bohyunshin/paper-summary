# RQ-VAE / RQ-Transformer 공부 정리

논문: *Autoregressive Image Generation using Residual Quantization*
(Lee et al., Kakao Brain/POSTECH, CVPR 2022, arXiv:2203.01941)

이 문서는 RQ-VAE 논문을 공부하며 주고받은 질문과 답을 순서대로 정리한 노트입니다.
수식 전개 과정과 핵심 개념을 모두 포함합니다.

---

## 0. 논문 개요

기존 VQ-VAE 계열은 autoregressive(AR) 모델로 고해상도 이미지를 생성할 때, 코드 시퀀스 길이를 줄이는 것과 reconstruction 품질을 유지하는 것이 서로 trade-off 관계였습니다. 이 논문은 두 단계로 구성된 프레임워크를 제안합니다.

- **Stage 1: RQ-VAE** — residual quantization(RQ)을 이용해 codebook 크기를 그대로 유지하면서 표현력을 끌어올리는 quantized autoencoder
- **Stage 2: RQ-Transformer** — RQ-VAE가 만든 D-depth 코드 스택을 효율적으로 예측하는 두 단(spatial + depth) transformer

---

## 1. VQ-VAE의 문제와 RQ-VAE의 motivation

### 질문: VQ-VAE가 구체적으로 어떤 문제가 있어서 RQ-VAE가 나온 거지? 어떤 motivation이 있는 거야?

**VQ-VAE의 구체적 문제**

VQ-VAE는 feature map의 각 위치 벡터를 codebook에서 가장 가까운 코드 하나로 양자화합니다.

$$
Q(z;C) = \arg\min_{k\in[K]} \Vert z - e(k) \Vert_2^2
$$

이미지를 표현하는 데 쓰는 정보량(rate)은 다음과 같습니다.

$$
HW \log_2 K
$$

정보량을 조절할 수 있는 손잡이는 두 개뿐입니다: 공간 해상도 $(H,W)$ 또는 codebook 크기 $K$.

1. **Rate-distortion trade-off가 codebook 크기에만 의존하는 구조**: AR 모델은 시퀀스가 짧아야(=$(H,W)$가 작아야) 학습이 빠르고 long-range dependency를 잘 잡습니다. 그런데 $(H,W) \to (H/2, W/2)$로 줄이면서 reconstruction 품질을 유지하려면 codebook을 $K^4$로 늘려야 합니다. 해상도를 절반으로 줄이는 대가가 codebook 크기의 네제곱입니다.
2. **Codebook을 키우면 codebook collapse**: $K$를 키우면 소수의 코드만 계속 선택되고 나머지는 거의 안 쓰이는 codebook collapse 문제가 생기고, 학습이 불안정해집니다.
3. **결과적으로 둘 다 가질 수 없음**: 짧은 시퀀스를 원하면 codebook이 터지고, codebook을 작게 유지하면 시퀀스가 길어져야 합니다.

**RQ-VAE의 motivation: 같은 정보량을 다른 축으로 분배**

발상의 전환: codebook 크기를 늘리는 대신, 같은 codebook을 여러 번 재사용한다.

벡터 하나를 코드 1개가 아니라 depth $D$개의 코드 묶음으로 표현하면,

- VQ가 만드는 클러스터 수: $K$
- depth $D$짜리 RQ가 만드는 클러스터 수: 최대 $K^D$

즉 codebook 자체의 크기는 그대로 $K$인데 표현력은 $K^D$로 지수적으로 늘어납니다. 정보량을 늘리는 손잡이가 $(H,W)$, $K$, 그리고 새로 추가된 $D$ 세 개가 됩니다. $D$를 늘리는 비용은 codebook collapse를 유발하지 않습니다(codebook 크기 자체는 안 바뀌므로).

**실험적 증거 (Table 4)**: $K=16384$로 고정한 채 $D$만 늘렸을 때 reconstruction FID(rFID)가 급격히 좋아집니다.

| $D$ | rFID |
| --- | --- |
| 1 (=VQ) | 17.95 |
| 4 | 4.73 |
| 8 | 2.69 |
| 16 | 1.83 |

**한 문장 요약**: VQ-VAE는 짧은 시퀀스와 정밀한 복원을 동시에 얻으려면 codebook을 지수적으로 키워야 했고 그게 collapse를 유발했다. RQ-VAE는 codebook 크기는 고정한 채 같은 codebook을 depth 방향으로 재사용해서 표현력을 지수적으로 늘림으로써, collapse 없이 짧은 시퀀스 + 정밀한 복원을 동시에 얻어낸다.

---

## 2. AR(Autoregressive)이라는 용어의 의미

### 질문: 자꾸 AR이라는 말이 나오는데 이게 정확히 어떤 의미야?

AR은 Autoregressive(자기회귀)의 줄임말입니다. 전체 시퀀스의 확률을, 이전 토큰들을 조건으로 다음 토큰 하나씩 예측하는 체인으로 분해하는 모델링 방식입니다.

식(9)가 정확히 이 구조입니다.

$$
p(\mathbf{S}) = \prod_{t=1}^{T} \prod_{d=1}^{D} p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right)
$$

말로 풀면: 지금까지 생성한 모든 코드들을 보고 다음 코드 하나를 예측한다, 를 시퀀스 끝까지 반복합니다. GPT가 텍스트를 생성하는 방식(이전 단어들을 보고 다음 단어 하나씩 예측)과 동일한 패러다임이며, 여기서는 단어가 아니라 이미지 코드를 다룹니다.

**왜 이미지에 AR을 쓰는가**

이미지를 직접 픽셀 단위로 AR하면 $256\times256\times3$개의 토큰을 순서대로 예측해야 해서 비효율적입니다. 그래서:

1. VQ/RQ-VAE로 이미지 → discrete code map으로 압축 (예: $8\times8\times D$)
2. 그 코드 시퀀스에 대해 AR 모델(Transformer)을 학습
3. 생성 시 코드를 하나씩 순차적으로 샘플링한 뒤 decoder $G$로 이미지 변환

AR 모델이 직접 다루는 대상은 픽셀이 아니라 VQ/RQ가 만든 코드 시퀀스입니다. RQ-VAE가 시퀀스 길이 $T=HW$를 줄여줄수록 AR 모델이 예측해야 할 스텝 수가 줄고 long-range dependency 학습도 쉬워집니다.

**이 논문에서 AR이 두 번 나오는 지점**

- Spatial Transformer: 위치 $t$에 대해 AR — 이전 위치들 $S_{<t}$를 보고 현재 위치의 context를 만듦
- Depth Transformer: 같은 위치 안에서 depth $d$에 대해 AR — 이미 뽑은 depth $1,\dots,d-1$ 코드를 보고 depth $d$ 코드를 예측

두 단계 모두 이전 것을 보고 다음 것 하나씩 순차적으로 예측한다는 점에서 동일하게 AR이며, 분해 축이 (공간 위치) vs (같은 위치 안의 depth)로 다를 뿐입니다.

---

## 3. Residual Quantization(RQ)의 구체적 메커니즘

### 질문: 어떤 벡터에 대해서 하나의 코드로 양자화하는 게 아니라 D개의 코드로 양자화한다는 거지? 그리고 잔차에 대해서 계속 양자화하는 거고

정확한 이해이며, 이 절에서는 메커니즘을 한 단계씩 전개합니다.

**기본 구조 (식 3, 4)**

벡터 $z \in \mathbb{R}^{n_z}$ 하나를 다음과 같이 순서가 있는 $D$개의 코드 튜플로 표현합니다.

$$
\mathrm{RQ}(z; C, D) = (k_1, k_2, \dots, k_D) \in [K]^D
$$

이 튜플을 만드는 재귀 절차는 다음과 같습니다.

$$
r_0 = z
$$

$$
k_d = Q(r_{d-1}; C)
\qquad
r_d = r_{d-1} - e(k_d)
\qquad
d = 1, \dots, D
$$

여기서 $Q(\cdot;C)$는 일반적인 VQ — codebook에서 가장 가까운 코드 하나를 고르는 연산입니다.

**손으로 따라가 보기**

- $d=1$: $r_0=z$ 자체를 양자화 → 가장 가까운 코드 $k_1$ 선택. $r_1 = z - e(k_1)$ (1차 근사로 못 잡은 나머지)
- $d=2$: 나머지 $r_1$을 다시 양자화 → $k_2 = Q(r_1; C)$. $r_2 = r_1 - e(k_2)$
- $d=3$: $r_2$를 양자화 → $k_3$. 계속 반복
- 이를 $D$번 반복

매 단계는 이전 단계까지 설명하지 못한 오차를 codebook으로 다시 근사하는 것입니다. VQ를 한 번이 아니라 잔차에 대해 재귀적으로 $D$번 거는 것입니다.

**부분합과 근사 정밀도**

$d$번째까지의 코드 임베딩 합을 부분합이라고 합니다.

$$
\hat{z}^{(d)} = \sum_{i=1}^{d} e(k_i)
$$

전개해 보면,

$$
\hat{z}^{(d)} = e(k_1) + e(k_2) + \cdots + e(k_d)
$$

$$
z - \hat{z}^{(d)} = z - \sum_{i=1}^{d} e(k_i) = r_d
$$

즉 $d$단계까지의 근사 오차가 정확히 $r_d$입니다. $\hat{z}^{(d)}$는 $d$가 커질수록 $z$에 더 가까워지며, 이를 coarse-to-fine 근사라고 합니다. $\hat{z}^{(1)}$은 codebook 안에서 $z$에 가장 가까운 점 하나(거친 근사)이고, $\hat{z}^{(D)}$는 그것을 $D$개의 codebook 벡터의 합으로 정교하게 다듬은 것입니다.

**왜 codebook 하나 재사용이 $K^D$ 표현력을 주는가**

각 단계 $k_d$는 $K$개의 codebook 중 아무거나 고를 수 있으므로(실제 선택은 이전 잔차에 의존하지만 가능한 조합의 수는 독립적), 튜플 $(k_1,\dots,k_D)$가 만들 수 있는 조합의 수는

$$
K \times K \times \cdots \times K = K^D
$$

이 $K^D$개의 서로 다른 조합 각각이 $\mathbb{R}^{n_z}$ 공간의 서로 다른 한 점($\hat{z}^{(D)}$의 값)에 대응합니다. codebook 크기는 $K$로 고정했지만 표현 가능한 최종 양자화 벡터의 개수는 $K^D$입니다.

**주의할 점: greedy이지 globally optimal이 아님**

이 알고리즘은 greedy입니다. 매 단계 $k_d$는 그 단계의 잔차를 최소화하는 코드를 고르는 것일 뿐, $D$개를 동시에 최적화해서 $\Vert z - \hat{z}^{(D)} \Vert$를 전역적으로 최소화하는 것이 아닙니다. (Related Work에서 언급된 additive quantization(AQ)이 NP-hard인 이유가 바로 이 전역 최적화를 시도하기 때문입니다.) RQ는 전역 최적화를 포기하고 순차적 greedy 근사로 타협한 대신, 계산이 가볍고($D$번의 NN search) 학습이 안정적이라는 실용적 이득을 얻습니다.

**Codebook 공유의 의미**

$d=1,\dots,D$ 모든 단계가 같은 $C$를 사용합니다. $k_1$을 고를 때 쓰는 codebook 벡터들과 $k_3$을 고를 때 쓰는 codebook 벡터들이 풀(pool)로는 동일하지만, 입력으로 들어가는 잔차의 통계적 분포(스케일, 분산)는 $d$가 커질수록 달라집니다(잔차가 점점 작아지고 더 미세한 디테일을 담음). 학습이 끝나면 같은 codebook 안에서도 주로 $d=1$에서 선택되는 코드들과 주로 $d=4$에서 선택되는 코드들이 암묵적으로 다른 역할(coarse structure vs fine detail)을 하게 됩니다.

---

## 4. RQ-VAE: VAE 구조 안에 RQ를 끼워 넣기

### 질문: 그러면 RQ-VAE는 뭐야?

RQ 자체는 벡터 하나를 어떻게 양자화하느냐에 대한 알고리즘이고, RQ-VAE는 이 RQ를 VAE 구조 안에 끼워 넣은 전체 모델입니다.

**VQ-VAE 구조**

$$
\text{Encoder } E \rightarrow \text{Quantization (VQ)} \rightarrow \text{Decoder } G
$$

1. 인코더 $E$가 이미지 $X \in \mathbb{R}^{H_o \times W_o \times 3}$를 받아서 feature map $Z = E(X) \in \mathbb{R}^{H \times W \times n_z}$로 압축
2. $Z$의 각 위치 $(h,w)$의 벡터를 VQ로 양자화 → 코드맵 $M \in [K]^{H \times W}$, 양자화된 feature map $\hat{Z}$
3. 디코더 $G$가 $\hat{Z}$를 받아서 이미지를 복원 $\hat{X} = G(\hat{Z})$

**RQ-VAE는 같은 구조에서 양자화 단계만 교체**

$$
\text{Encoder } E \rightarrow \text{Quantization (RQ)} \rightarrow \text{Decoder } G
$$

feature map $Z$의 각 위치 $(h,w)$의 벡터에 대해 코드 하나가 아니라 $D$개의 코드 튜플을 뽑습니다.

$$
M_{hw} = \mathrm{RQ}\left(E(X)_{hw}; C, D\right)
$$

전체 코드맵은 $M \in [K]^{H \times W \times D}$가 됩니다(기존 VQ-VAE의 코드맵 $[K]^{H \times W}$에 깊이 축 $D$가 추가됨).

각 depth $d$까지의 부분합으로 만든 양자화된 feature map은

$$
\hat{Z}^{(d)}_{hw} = \sum_{d'=1}^{d} e(M_{hwd'})
$$

최종적으로 $\hat{Z} := \hat{Z}^{(D)}$를 디코더에 넣어 복원합니다: $\hat{X} = G(\hat{Z})$.

**비교표**

| | VQ-VAE | RQ-VAE |
| --- | --- | --- |
| 인코더/디코더 | $E$, $G$ | 동일 |
| 위치 하나당 코드 수 | 1개 | $D$개 (depth) |
| 코드맵 shape | $[K]^{H \times W}$ | $[K]^{H \times W \times D}$ |
| 양자화 연산 | $Q(z;C)$ 한 번 | $\mathrm{RQ}(z;C,D)$ — $Q$를 잔차에 D번 반복 |

**한 줄 요약**: RQ-VAE는 VQ-VAE의 양자화 단계를 단일 코드 선택에서 잔차 기반 D-step 코드 선택(RQ)으로 바꾼 것입니다. 인코더-디코더는 그대로이며, 사이에 끼워진 quantizer만 RQ로 바뀌었습니다. 그 결과로 codebook collapse 없이 $K^D$ 표현력을 얻는다는 이득이 통째로 모델에 적용됩니다.

---

## 5. RQ-VAE의 Loss와 VQ-VAE 대비 항의 차이

### 질문: RQ-VAE는 VQ-VAE의 loss 중에 첫번째, 세번째 텀을 사용하는 것으로 보여. 차이점이 뭐야? RQ-VAE는 왜 이 loss를 택했을까?

VQ-VAE의 원래 loss는 보통 세 항으로 적습니다.

$$
\mathcal{L}_{VQVAE} =
\Vert X - \hat{X} \Vert_2^2
+ \Vert sg[Z] - \hat{Z} \Vert_2^2
+ \beta \Vert Z - sg[\hat{Z}] \Vert_2^2
$$

순서대로 reconstruction loss, codebook loss, commitment loss입니다. RQ-VAE는 이 중 1번째(reconstruction)와 3번째(commitment)만 쓰고 2번째 codebook loss를 뺀 형태입니다.

**왜 2번째 항이 빠졌는가 — codebook 업데이트 방식이 다르기 때문**

세 항의 원래 역할:

- Reconstruction loss: 인코더 $E$와 디코더 $G$를 학습시키는 신호
- Codebook loss $\Vert sg[Z] - \hat{Z} \Vert_2^2$: codebook 임베딩 $e(k)$ 자체를 $Z$ 쪽으로 끌어당기는 신호 (gradient가 $e(k)$로만 흐르도록 $Z$에 stop-gradient)
- Commitment loss $\Vert Z - sg[\hat{Z}] \Vert_2^2$: 반대로 인코더 출력 $Z$를 codebook 쪽으로 끌어당기는 신호 (gradient가 $E$로만 흐르도록 $\hat{Z}$에 stop-gradient)

codebook loss와 commitment loss는 원래 서로 마주보는 두 개의 끌어당기는 힘입니다. 코드북을 인코더 출력 쪽으로, 동시에 인코더 출력을 코드북 쪽으로 당겨서 둘이 수렴하게 만듭니다.

RQ-VAE는 codebook $C$를 gradient descent로 업데이트하지 않고 EMA(exponential moving average)로 업데이트합니다(VQ-VAE 원 논문에 나온 대안적 업데이트 방식을 채택). EMA 업데이트는 codebook 벡터를 그 코드를 선택한 인코더 출력들의 이동평균으로 직접 갱신하므로, gradient 기반 codebook loss가 더 이상 필요하지 않습니다.

RQ-VAE의 최종 loss:

$$
\mathcal{L} = \mathcal{L}_{recon} + \beta \mathcal{L}_{commit}
$$

$$
\mathcal{L}_{commit} = \sum_{d=1}^{D} \Vert Z - sg[\hat{Z}^{(d)}] \Vert_2^2
$$

codebook loss가 빠진 이유는 손실 함수가 단순해서가 아니라, codebook 업데이트의 책임을 loss term에서 EMA 알고리즘으로 이전했기 때문입니다(VQ-VAE 원 논문에도 있던 두 옵션 중 하나를 채택).

**왜 RQ-VAE는 EMA를 택했을까**

1. **Commitment loss가 이미 D개 항의 합이라 gradient 흐름이 복잡함**: 각 depth 근사에 대해 인코더가 책임을 지도록 강제하는 항인데, 여기에 codebook loss까지 $D$개 항으로 추가하면 각 depth의 codebook 벡터들이 서로 다른 스케일(잔차 크기는 depth가 깊어질수록 작아짐)을 가진 타깃으로 동시에 끌려가게 되어 gradient 기반 업데이트가 불안정해지기 쉽습니다.
2. **Shared codebook 설계와 맞물림**: 모든 depth가 같은 codebook $C$를 공유하는데, depth마다 입력 잔차의 통계(분산, 스케일)가 다릅니다. 하나의 codebook이 스케일이 다른 입력 분포 전체를 잘 커버하려면, 각 코드가 그 코드를 선택한 모든 (depth, 위치) 쌍의 평균적인 위치로 안정적으로 자리잡아야 하는데, EMA가 이런 다양한 소스의 클러스터 중심을 안정적으로 추적하는 데 gradient descent보다 유리합니다(online k-means에 가까움).
3. **VQ-VAE 계열에서 EMA가 이미 더 안정적이라고 알려져 있었음**: EMA 업데이트가 gradient 기반 codebook loss보다 codebook이 더 골고루 쓰이고(collapse가 덜함) 학습이 안정적이라는 게 알려져 있었습니다. RQ-VAE의 핵심 motivation이 codebook collapse 회피였으므로, codebook 업데이트 메커니즘부터 더 안정적인 쪽(EMA)을 선택하는 것이 일관된 선택입니다.

**비교표**

| | VQ-VAE (gradient 버전) | RQ-VAE |
| --- | --- | --- |
| Codebook 업데이트 | codebook loss의 gradient | EMA (클러스터 평균 추적) |
| Loss에 codebook loss 필요? | 필요 | 불필요 (EMA가 대체) |
| Commitment loss | 단일 항 | depth 합 ($D$개 항) |

빠진 항은 버린 것이 아니라 다른 메커니즘(EMA)으로 역할을 옮긴 것이고, 이 선택은 RQ 구조의 핵심 동기(codebook collapse 회피, shared codebook의 다양한 입력 스케일 처리)와 자연스럽게 맞물립니다.

---

## 6. RQ-VAE와 RQ-Transformer의 관계

### 질문: RQ-Transformer에 대해서 이해하고 싶어. 이거는 RQ-VAE와 따로 학습하는 거야? Multi-task learning은 아니고? 이 둘의 관계는 뭐야? 왜 VQ-VAE에는 Transformer 개념이 안 나온 거야?

**(1) RQ-VAE와 RQ-Transformer는 따로 학습 (multi-task 아님)**

논문은 명시적으로 two-stage framework라고 밝힙니다.

- **Stage 1**: RQ-VAE를 먼저 끝까지 학습 ($\mathcal{L}_{recon} + \beta \mathcal{L}_{commit}$). 학습이 끝나면 인코더 $E$, 디코더 $G$, codebook $C$가 모두 고정됩니다.
- **Stage 2**: Stage 1에서 학습이 끝나고 고정된 RQ-VAE를 이용해 전체 학습 데이터셋의 이미지를 코드 시퀀스로 변환합니다. 이 코드 시퀀스로 RQ-Transformer를 처음부터 별도로 학습합니다 ($\mathcal{L}_{AR}$).

실험에서 하나의 RQ-VAE를 고정해 두고 여러 RQ-Transformer를 갈아 끼우는 사례(ImageNet에서 학습한 RQ-VAE를 재사용해서 CC-3M용 RQ-Transformer를 학습)가 나오는데, 이는 두 모델이 별개로 학습된다는 직접적인 증거입니다. Multi-task learning이었다면 이런 재사용이 말이 안 됩니다.

같이 학습시키지 않는 이유:

- RQ-VAE의 목표(픽셀 단위 reconstruction)와 RQ-Transformer의 목표(코드 시퀀스의 likelihood)는 서로 다른 종류의 최적화 문제라서, 같이 학습시키면 quantizer가 계속 흔들리는(non-stationary target) 상태에서 AR 모델이 분포를 따라가야 하는 어려움이 생깁니다.
- VQ-VAE 계열 전체가 먼저 discrete 표현을 안정적으로 굳히고, 그 다음에 그 위에서 생성 모델을 학습하는 2-stage 패러다임을 따릅니다 (VQ-VAE2, VQ-GAN, DALL-E 1세대 모두 동일).

**(2) 둘의 관계 — 압축기와 그 압축 공간 위에서 동작하는 생성기**

$$
X \xrightarrow{E} Z \xrightarrow{\mathrm{RQ}} M \in [K]^{H \times W \times D} \xrightarrow{\text{reshape}} S \in [K]^{T \times D}
$$

$$
S \xrightarrow{\text{RQ-Transformer가 } p(S) \text{ 모델링}} S_{\text{생성}} \xrightarrow{e(\cdot), G} X_{\text{생성}}
$$

- RQ-VAE는 이미지 $X$를 코드 시퀀스 $S$로 바꾸는(거꾸로 $S$를 다시 이미지로 바꾸는) 변환기/압축기입니다. 생성 능력은 없습니다.
- RQ-Transformer는 그 코드 시퀀스 $S$의 분포 $p(S)$를 학습하는 생성 모델입니다. 학습 후에는 실제 이미지 없이도 $p(S)$에서 새로운 코드 시퀀스를 샘플링하고, RQ-VAE의 디코더에 넣어 새 이미지를 만들 수 있습니다.

비유: RQ-VAE는 언어(discrete code vocabulary)를 정의하고 텍스트를 그 언어로 번역/역번역하는 역할이고, RQ-Transformer는 그 언어로 새로운 문장을 짓는 역할입니다. 다만 여기서는 vocabulary 자체(codebook)를 RQ-VAE가 데이터로부터 학습해서 만들어낸다는 점이 일반 언어와 다릅니다.

**(3) 왜 VQ-VAE 논문엔 Transformer가 없었나**

VQ-VAE(van den Oord 2017) 원 논문 자체는 이미지를 discrete code로 어떻게 잘 압축하느냐만 다룬 연구였고, 그 코드 위에서 생성을 어떻게 하느냐는 별도 모델(PixelCNN)로 다뤘습니다. 즉 VQ-VAE 시절부터 이미 quantizer + 별도의 AR prior라는 2단 구조였고, Transformer가 안 나온 건 그 시점(2017년, Transformer 논문이 같은 해 발표)에 Transformer가 아직 이 분야의 기본 선택지가 아니었기 때문입니다.

시대별 흐름:

| 모델 | Stage 1 (quantizer) | Stage 2 (AR prior) |
| --- | --- | --- |
| VQ-VAE (2017) | VQ-VAE | PixelCNN |
| VQ-VAE-2 (2019) | hierarchical VQ-VAE | PixelCNN |
| VQ-GAN (2021) | VQ-VAE + adversarial loss | Transformer (GPT 스타일) |
| RQ-VAE (2022) | RQ-VAE | RQ-Transformer |

Transformer를 쓰는 것은 VQ-GAN 시점부터 이미 표준이 된 선택이고, RQ-VAE 논문은 여기에 자기만의 새로운 quantizer(RQ)와, 그 quantizer가 만든 D-depth 코드 구조에 맞춘 Transformer 변형(spatial+depth 분리)을 제안한 것입니다. RQ-Transformer라는 이름이 따로 붙은 이유는 일반 GPT 구조 그대로는 $D$ 차원이 추가된 코드맵을 효율적으로 다룰 수 없어서(아래 7절의 $O(NT^2D^2)$ 문제), spatial transformer + depth transformer로 쪼갠 전용 아키텍처를 새로 설계했기 때문입니다.

**한 줄 요약**: VQ-VAE는 quantizer 자체에 대한 연구였고 AR 모델은 항상 별도였습니다(PixelCNN→Transformer로 시대에 따라 교체됨). RQ-VAE도 그 전통을 따르되, 자신의 D-depth 코드 구조에 최적화된 전용 Transformer(RQ-Transformer)를 별도로 새로 설계해서 짝지었고, 둘은 순차적으로 독립적으로 학습됩니다.

---

## 7. 식(8), (9) — 코드맵의 reshape과 AR factorization

### 질문: 식(8)과 (9)가 잘 이해가 안 가는데, 구체적인 예시를 들어서 설명해줘

**설정값 정하기**

- $H=2, W=2$ → $T = HW = 4$
- $D=3$
- codebook 코드는 알파벳으로 표기 ($a, b, c, \dots$)

**1단계: $\mathbf{M} \in [K]^{H \times W \times D}$ — 원래 코드맵**

```text
위치(0,0): (a, x, p)        위치(0,1): (b, y, q)
위치(1,0): (c, z, r)        위치(1,1): (d, w, s)
```

shape은 $2 \times 2 \times 3 = H \times W \times D$.

**2단계: raster scan으로 $\mathbf{S} \in [K]^{T \times D}$로 펴기**

Raster scan은 왼쪽 위 → 오른쪽, 그 다음 줄로 내려가서 또 왼쪽 → 오른쪽 순서로 2D 격자를 1D로 펴는 것입니다.

$$
t=1: (0,0) \quad t=2: (0,1) \quad t=3: (1,0) \quad t=4: (1,1)
$$

$\mathbf{S}$는 $T \times D = 4 \times 3$ 행렬이 됩니다.

| $t$ | $S_{t1}$ | $S_{t2}$ | $S_{t3}$ |
| --- | --- | --- | --- |
| $t=1$ | $a$ | $x$ | $p$ |
| $t=2$ | $b$ | $y$ | $q$ |
| $t=3$ | $c$ | $z$ | $r$ |
| $t=4$ | $d$ | $w$ | $s$ |

이것이 식(8)의 내용입니다.

$$
\mathbf{S}_t = (S_{t1}, \dots, S_{tD}) \in [K]^D \quad \text{for } t \in [T]
$$

$\mathbf{S}_t$는 위치 $t$에 해당하는 한 행(depth $D$개 코드)이고, 이게 $t=1,\dots,T$ 전체에 대해 쌓이면 $\mathbf{S}$ 행렬이 됩니다.

**3단계: 식(9)를 풀어 쓰기**

$$
p(\mathbf{S}) = \prod_{t=1}^{T} \prod_{d=1}^{D} p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right)
$$

이 식은 $T \times D = 4 \times 3 = 12$개의 코드 전체를 정해진 순서로 하나씩 조건부 확률의 곱으로 분해하겠다는 선언입니다. 그 순서는 이중 for문 그대로입니다.

```text
바깥 루프: t = 1 → 4   (공간 위치, raster scan 순서)
  안쪽 루프: d = 1 → 3  (같은 위치 안에서 depth 순서)
```

실제로 예측되는 순서:

$$
S_{11} \to S_{12} \to S_{13} \to S_{21} \to S_{22} \to S_{23} \to S_{31} \to \cdots \to S_{43}
$$

즉 $a \to x \to p \to b \to y \to q \to c \to z \to r \to d \to w \to s$ 순서로 12개 코드를 하나씩 예측합니다.

**조건부 항 풀어보기 — $t=2, d=2$ ($y$를 예측하는 순간)**

$$
p\left(S_{22} \mid S_{<2,2}, S_{2,<2}\right)
$$

- $S_{<2,2}$: 이전 위치($t<2$, 즉 $t=1$)의 모든 코드 → $(a, x, p)$ 전체
- $S_{2,<2}$: 같은 위치($t=2$)에서 이미 예측한 더 낮은 depth → $d<2$이므로 $S_{21}=b$만

풀어 쓰면,

$$
p\left(y \mid a, x, p, b\right)
$$

위치 1의 코드 3개를 다 보고, 위치 2 안에서는 depth 1 코드($b$)까지 본 다음 위치 2의 depth 2 코드($y$)를 예측한다는 뜻입니다.

**전체를 곱으로 펼치면**

$$
\begin{aligned}
p(\mathbf{S}) =
& p(a)\ p(x \mid a)\ p(p \mid a, x) \\
&\times p(b \mid a, x, p)\ p(y \mid a, x, p, b)\ p(q \mid a, x, p, b, y) \\
&\times \cdots
\end{aligned}
$$

매 항이 지금까지 나온 모든 코드를 빠짐없이 다 본 다음, 다음 코드 하나를 예측하는 chain rule입니다. 다만 이 지금까지 나온 모든 코드를 두 그룹(다른 위치에서 온 것 $S_{<t,d}$, 같은 위치에서 온 것 $S_{t,<d}$)으로 갈라서 표기한 것입니다.

**왜 두 인덱스로 나눠 적었는가**

이게 다음 절(RQ-Transformer 아키텍처)로 가는 이유입니다 — 이 둘이 서로 다른 transformer가 처리하기 때문에 미리 구분해놓은 것입니다.

- $S_{<t,d}$ (다른 위치들 전체) → Spatial Transformer가 처리해서 context vector $h_t$로 압축
- $S_{t,<d}$ (같은 위치, 더 낮은 depth) → Depth Transformer가 $h_t$를 받은 다음 그 위치 안에서만 순차적으로 처리

---

## 8. 두 transformer로 나눈 이유 — 계산 복잡도

### 질문: Spatial transformer, depth transformer로 나눠서 한 이유는 계산 때문인거지?

핵심 동기는 계산량입니다.

**문제: naïve하게 하면 시퀀스 길이가 $T \times D$**

12개 코드를 1차원으로 펴서 일반 GPT 스타일 transformer 하나에 통째로 넣으면 시퀀스 길이는 $TD$가 되고, self-attention 비용은 시퀀스 길이의 제곱이므로

$$
O\left(N \cdot (TD)^2\right) = O\left(N T^2 D^2\right)
$$

RQ-VAE의 핵심 동기가 $T$(시퀀스 길이)를 줄이는 것이었는데, $D$를 추가로 도입해서 표현력을 늘렸더니 AR 모델 쪽 비용이 $D^2$만큼 다시 늘어나는 역설이 생깁니다.

**해결: $T$와 $D$를 같은 시퀀스 축에 섞지 않는다**

- Spatial Transformer — 시퀀스 길이 $T$, 위치 $t$들 사이의 attention만 계산:

$$
O\left(N_{spatial} T^2\right)
$$

- Depth Transformer — 시퀀스 길이 $D$(훨씬 짧음), 한 위치 안에서 depth $1,\dots,D$ 사이의 attention만 계산. 위치마다 $T$번 실행되므로:

$$
O\left(N_{depth} T D^2\right)
$$

합치면,

$$
O\left(N_{spatial} T^2 + N_{depth} T D^2\right)
$$

이는 $O(NT^2D^2)$보다 압도적으로 작습니다. $T^2D^2$의 $D^2$가 $TD^2$로(제곱이 아니라 1차로) 떨어지는 것이 핵심입니다.

**숫자로 감 잡기**

$T = 8 \times 8 = 64$, $D=4$로 넣으면,

- Naïve: $T^2D^2 = 64^2 \times 4^2 = 4096 \times 16 = 65536$
- 분리: $T^2 + TD^2 = 4096 + 64 \times 16 = 4096 + 1024 = 5120$

약 12배 이상 차이가 나며, $D$가 커질수록 차이는 더 벌어집니다.

**단순 트릭이 아닌 이유**

이 분리는 계산 절약과 동시에 식(9)의 조건부 구조와도 정확히 맞아떨어집니다.

- $S_{<t,d}$ (다른 위치 전체): 위치 $t$ 시점에서 이미 끝까지(depth $D$까지) 확정된 과거 정보 — 한 번 요약해서 context vector $h_t$ 하나로 압축해도 됨 (Spatial Transformer)
- $S_{t,<d}$ (같은 위치, 낮은 depth): 지금 막 만들어지고 있는, 아직 끝나지 않은 현재 위치 내부의 순서 — 위치마다 새로 짧게 풀어줘야 함 (Depth Transformer)

식(9)의 factorization 자체에 내재된 (위치, depth)라는 2단 계층 구조를 아키텍처에 그대로 반영한 것이고, 그 결과로 계산량도 같이 줄어듭니다.

---

## 9. 정보 손실에 대한 추가 논의

### 질문: 이게 정보손실과 무슨 관련이 있는 거야?

**비교 기준: 이상적인 full attention**

식(9)가 요구하는 것은 $S_{td}$를 예측할 때 이전에 나온 모든 코드($S_{<t,d}$와 $S_{t,<d}$ 둘 다)를 전부 조건으로 보는 것입니다. 이를 가장 충실하게 구현하려면 12개 코드를 일렬로 펴서 하나의 transformer에 통째로 넣고 self-attention을 거는 것(naïve 방식, 비용 $O(NT^2D^2)$)이 가장 직접적입니다.

**분리했을 때 정보가 줄어들 수 있는 두 가지 위험**

- 위험 1 — 압축 손실: Spatial Transformer가 위치 $t$ 이전의 모든 정보($S_{<t,d}$, $(t-1) \times D$개의 코드)를 단 하나의 벡터 $h_t$로 압축. 압축이 정보를 일부 버리면 depth transformer는 $S_{<t,d}$ 전체를 못 보고 요약본만 보게 됨.
- 위험 2 — 잘못 쪼개기: (위치, depth) 축을 무시하고 임의의 길이 $T$ 단위로 끊었다면 청크 경계에서 조건부 관계가 깨짐.

**RQ-Transformer가 정보 손실 없다고 한 이유**

위험 2는 발생하지 않습니다 — 쪼개는 기준이 식(9) 자체가 이미 가지고 있던 (위치, depth)라는 자연스러운 경계이기 때문입니다. 식(9)의 조건이 정확히 $S_{<t,d}$(다른 위치)와 $S_{t,<d}$(같은 위치, 낮은 depth) 두 그룹으로 나뉘어 있고, 두 transformer의 역할 분담이 이 두 그룹과 1:1로 대응합니다.

위험 1(압축 손실)은 실제로 존재하며 의도된 trade-off입니다. $h_t$가 $S_{<t,d}$ 전체를 손실 없이 보존한다는 보장은 없습니다 — 이는 모든 RNN/Transformer 기반 언어모델이 공통으로 갖는 한계입니다(GPT도 attention으로 과거를 직접 보지만, 과거 토큰들의 hidden representation 자체가 이미 압축된 표현). 여기서 정보 손실 없다는 것은 naïve full-attention 버전과 수학적으로 완전히 동일한 결과를 보장한다는 뜻이 아니라, 식(9)의 조건부 분해 구조를 깨뜨리지 않고 그대로 따른다는 뜻입니다.

**정정 및 정리**

- 구조적으로 보존되는 것: 식(9)가 정의한 조건부 의존성의 그래프 구조(누가 누구를 조건으로 삼는가) — 두 transformer로 쪼개도 그대로 유지됨. depth transformer가 보는 $h_t$는 위치 $t$ 이전 전체를 대표하는 역할을 정확히 수행하고, $S_{t,<d}$는 depth transformer 내부에서 명시적으로 그대로 입력됨 (식 12).
- 근사로 들어가는 것: $h_t$라는 고정 크기 벡터가 위치 $t$ 이전 전체라는 가변적이고 방대한 정보를 얼마나 잘 담아내느냐는 모델의 표현력과 학습 품질에 달린 문제 — 이는 naïve 버전이든 분리 버전이든 모든 autoregressive 모델이 공통으로 안고 가는 근사이며, 분리했기 때문에 새로 생긴 손실이 아님.

**정리**: 두 transformer로 쪼갠 것 자체는 식(9)의 분해 구조를 깨지 않는, 구조적으로 동등한 재구성입니다. 정보 손실이라고 부를 수 있는 부분은 $h_t$의 압축 표현력 한계인데, 이는 분리 설계 때문에 새로 생긴 것이 아니라 애초에 모든 transformer 기반 AR 모델이 가진 일반적인 한계입니다.

---

## 10. Spatial Transformer — 식(10)의 $u_t$

### 질문: Spatial transformer 이해하고 싶어. 식(10)의 $u_t$가 의미하는 것이 뭐야?

식(10):

$$
u_t = PE_T(t) + \sum_{d=1}^{D} e(S_{t-1,d}) \qquad \text{for } t > 1
$$

**$u_t$가 의미하는 것: spatial transformer에 들어가는 위치 $t$의 입력 토큰**

GPT 스타일 transformer에서 위치 $t$의 입력은 보통 (위치 $t-1$에서 실제로 나온 토큰의 임베딩) + (위치 임베딩)입니다. $u_t$도 동일한 역할인데, 토큰 하나가 아니라 위치 하나에 D개의 코드가 스택되어 있다는 점이 다릅니다.

**둘째 항: $\sum_{d=1}^{D} e(S_{t-1,d})$**

직전 위치 $t-1$에서 나온 $D$개 코드의 임베딩을 전부 더한 것입니다. 예시($D=3$, 위치 $t-1$의 코드 스택이 $(a,x,p)$)라면

$$
\sum_{d=1}^{3} e(S_{t-1,d}) = e(a) + e(x) + e(p)
$$

이는 식(5)에서 본 부분합 $\hat{Z}^{(d)}$의 정의를 그대로 따른 것입니다.

$$
\hat{Z}^{(d)}_{hw} = \sum_{d'=1}^{d} e(M_{hwd'})
$$

$d=D$일 때, 즉 모든 depth를 다 더하면 이는 위치 $t-1$의 (양자화 후) 최종 feature vector $\hat{Z}_{t-1}$입니다. 즉

$$
\sum_{d=1}^{D} e(S_{t-1,d}) = \hat{Z}_{t-1}
$$

직전 위치의 (RQ-VAE가 만든) 양자화된 feature vector 그 자체입니다.

**첫째 항: $PE_T(t)$**

raster-scan 순서에서 지금 내가 몇 번째 위치인지를 알려주는 위치 임베딩(positional embedding)입니다.

**합쳐서 보면**

$$
u_t = \underbrace{PE_T(t)}_{\text{나는 } t \text{번째 위치다}} + \underbrace{\hat{Z}_{t-1}}_{\text{직전 위치의 완성된 내용}}
$$

$t$번째 위치의 spatial transformer 입력은, 직전 위치($t-1$)에서 RQ-VAE가 최종적으로 만들어낸 quantized feature vector에다 지금 위치가 몇 번째인지 정보를 더한 것입니다.

**$t=1$일 때는 다른 처리**

$t=1$은 raster scan에서 첫 번째 위치라서 $t-1=0$, 즉 직전 위치가 존재하지 않습니다. 그래서 $u_1$은 둘째 항을 쓸 수 없고, 대신 학습 가능한 별도의 시작 임베딩을 씁니다(`<SOS>` 역할).

**정리**

| 구성 요소 | 의미 |
| --- | --- |
| $u_t$ | spatial transformer가 위치 $t$에서 받는 입력 토큰 |
| $PE_T(t)$ | 지금 $t$번째 위치라는 위치 정보 |
| $\sum_{d=1}^{D} e(S_{t-1,d}) = \hat{Z}_{t-1}$ | 직전 위치에서 D개 코드를 전부 합친, 그 위치의 완성된 양자화 feature vector |
| $u_1$ | 예외적으로 학습 가능한 시작 토큰 (직전 위치가 없으므로) |

---

## 11. Spatial Transformer 출력 — 식(11)의 $h_t$

### 질문: 식(11)이 의미하는 바가 무엇일까? Spatial transformer가 그냥 일반 transformer와 다른 거야?

식(11):

$$
h_t = \mathrm{SpatialTransformer}(u_1, \dots, u_t)
$$

**Spatial transformer는 일반 transformer**

구조 자체는 특별하지 않습니다. masked self-attention을 쌓은 일반적인 decoder-only transformer입니다. 입력 $u_1,\dots,u_t$를 받아서 causal mask(미래 위치를 못 보게 막는 마스크)를 건 self-attention을 여러 층 통과시키고, 위치 $t$에 해당하는 출력을 $h_t$라고 부르는 것뿐입니다. spatial이라는 이름은 다루는 시퀀스 축이 공간 위치($t=1,\dots,T$)라서 붙은 명명이며, 뒤에 나올 depth transformer(같은 구조인데 depth 축 $d=1,\dots,D$를 다루는)와 구분하려는 것입니다.

**식(11)이 의미하는 것**

위치 1부터 $t$까지의 입력 토큰 $u_1,\dots,u_t$를 spatial transformer(causal self-attention)에 통과시켜서, 위치 $t$에서의 context vector $h_t$를 얻는다는 뜻입니다.

$u_1,\dots,u_{t-1}$ 각각은(식 10) 위치 $t-2$의 완성된 양자화 벡터 + 위치 임베딩이었습니다. $u_1,\dots,u_t$ 전체를 transformer에 넣는다는 것은, raster scan 순서로 지금까지 지나온 모든 위치들의 (RQ-VAE가 만든) 완성된 feature vector들을 전부 한 번씩 본다는 뜻입니다.

그 결과로 나온 $h_t$는 — causal self-attention의 일반적 성질대로 — 위치 1부터 $t$까지의 모든 정보를 요약한, $t$번째 위치에 대한 context vector입니다.

**정리**

| | |
| --- | --- |
| 모델 구조 자체 | 특별하지 않음 — masked self-attention을 쌓은 일반 GPT-style transformer |
| 특별한 부분 | 입력으로 넣는 $u_t$가 일반 토큰 임베딩이 아니라, RQ-VAE가 만든 D-depth 코드를 전부 합친 quantized feature vector라는 점 |
| 출력 $h_t$의 의미 | raster-scan 순서로 위치 $t$ 이전까지 RQ-VAE가 만들어낸 모든 정보를 압축한 context vector |
| 다음 단계와의 연결 | $h_t$가 depth transformer의 초기 입력으로 들어가서 그 위치의 D개 코드를 순차적으로 예측하는 데 쓰임 |

spatial transformer라는 이름은 아키텍처적 혁신이 아니라 역할 분담의 이름입니다 — transformer 자체는 표준이고, 혁신은 (1) D개 코드를 합쳐서 입력 토큰으로 쓴다는 점(식 10), (2) 그 출력을 depth transformer라는 별도 모듈에 넘겨준다는 점에 있습니다.

---

## 12. $h_t$의 개수

### 질문: 그럼 $h_t$도 $T$ ($H \times W$)개만큼 나오는 거지?

맞습니다. $t = 1, 2, \dots, T$ 각각에 대해 $h_t$가 하나씩 나오므로 총 $T$개의 context vector $h_1, h_2, \dots, h_T$가 생깁니다. $T = HW$였으므로 결국 공간상의 위치 하나당 $h_t$ 하나입니다.

**왜 $T$개가 나오는지**

causal self-attention을 쌓은 transformer는 입력 시퀀스 $u_1,\dots,u_T$ 전체를 한 번에 받아서 출력도 같은 길이로 내놓습니다 — 입력 토큰 하나당 출력 토큰(여기선 context vector) 하나가 대응되는 것이 transformer의 기본 동작입니다(causal mask 때문에 $h_t$는 $u_1,\dots,u_t$까지만 보고 만들어지지만, 위치 $t$마다 출력 슬롯은 하나씩 있습니다).

식(11)을 $t$마다 펼쳐 쓰면,

$$
h_1 = \mathrm{SpatialTransformer}(u_1)
$$

$$
h_2 = \mathrm{SpatialTransformer}(u_1, u_2)
$$

$$
\vdots
$$

$$
h_T = \mathrm{SpatialTransformer}(u_1, \dots, u_T)
$$

실제 구현에서는 $u_1,\dots,u_T$ 전체를 한 번의 forward pass로 넣고, causal mask가 위치 $t$의 출력은 $u_1,\dots,u_t$만 본다는 제약을 attention 단계에서 강제합니다. 한 번의 호출로 $h_1,\dots,h_T$ 전부가 동시에 나옵니다.

**도식**

```text
입력:   u_1   u_2   u_3   ...   u_T
         |     |     |           |
   [ Spatial Transformer (causal self-attn) ]
         |     |     |           |
출력:   h_1   h_2   h_3   ...   h_T
```

각 $h_t$는 다음 단계에서 위치 $t$의 depth transformer로 넘어가서, 그 위치의 $D$개 코드 $(S_{t1},\dots,S_{tD})$를 순차적으로 예측하는 데 쓰입니다.

$$
T\text{개의 } u_t
\xrightarrow{\text{spatial transformer (한 번)}}
T\text{개의 } h_t
\xrightarrow{\text{depth transformer (위치마다)}}
T \times D\text{개의 } S_{td}
$$

spatial transformer는 $T$ 길이 시퀀스를 한 번 처리해서 $T$개의 context를 만들고, 그 다음 depth transformer가 그 $T$개 각각에 대해 (짧은 길이 $D$짜리) 작업을 따로 수행하는 구조입니다 — 이는 앞서 본 $O(N_{spatial}T^2 + N_{depth}TD^2)$ 비용 분해와 정확히 일치합니다.

---

## 13. Depth Transformer — 식(12)와 식(10)의 차이

### 질문: Depth transformer도 이해해보자. 식(12)와 (10)의 차이점은?

식(10):

$$
u_t = PE_T(t) + \sum_{d=1}^{D} e(S_{t-1,d}) \qquad \text{for } t > 1
$$

식(12):

$$
v_{td} = PE_D(d) + \sum_{d'=1}^{d-1} e(S_{t,d'}) \qquad \text{for } d > 1
$$

**구조는 동일 — (위치 임베딩) + (이전 코드 임베딩들의 합)**

두 식 모두 형태가 같습니다. 차이는 모두 어느 축을 따라가느냐에서 나옵니다.

**차이점 1: 누적하는 축이 다르다**

| | 식(10) $u_t$ | 식(12) $v_{td}$ |
| --- | --- | --- |
| 누적 축 | 위치 $t$ (raster scan) | depth $d$ (같은 위치 안에서) |
| 합산 범위 | $\sum_{d=1}^{D}$ — 직전 위치의 모든 depth | $\sum_{d'=1}^{d-1}$ — 같은 위치의 더 낮은 depth까지만 |
| 가리키는 대상 | $S_{t-1,d}$ — 다른 위치($t-1$), 모든 $d$ | $S_{t,d'}$ — 같은 위치($t$), 낮은 $d'$ |

$u_t$의 합은 항상 $d=1$부터 $D$까지 다 끝난 상태($\hat{Z}_{t-1}$, 완성된 벡터)를 가져오는 반면, $v_{td}$의 합은 $d'=1$부터 $d-1$까지, 즉 아직 다 끝나지 않은 지금 만들어지고 있는 도중의 부분합 $\hat{Z}_t^{(d-1)}$을 가져옵니다.

$$
\sum_{d'=1}^{d-1} e(S_{t,d'}) = \hat{Z}_t^{(d-1)}
$$

$v_{td}$의 둘째 항은 같은 위치 $t$에서 depth $d-1$까지만 본 중간 근사입니다 — RQ가 잔차를 깎아나가던 그 부분합이 다시 등장하는 것입니다.

**차이점 2: 위치 임베딩이 가리키는 것이 다르다**

$PE_T(t)$는 공간상 $T$개 위치 중 몇 번째인지, $PE_D(d)$는 depth $D$단계 중 몇 번째인지 — 서로 다른 축의 좌표를 인코딩합니다.

논문은 $v_{td}$에 $PE_T(t)$를 넣지 않는 이유를 직접 밝힙니다.

> "We do not use $PE_T(t)$ in $v_{td}$, since the positional information is already encoded in $u_t$."

위치 정보($t$)는 이미 $h_t$를 통해 depth transformer로 전달되어 있으므로, $v_{td}$에서는 depth 위치($d$)만 새로 알려주면 충분합니다.

**차이점 3: $d=1$일 때 처리가 다르다**

식(10)은 $t=1$일 때 학습 가능한 시작 임베딩을 따로 둔다고 했습니다. 식(12)는 $d=1$일 때 spatial transformer의 출력 $h_t$를 직접 가져다 씁니다.

$$
v_{t1} = PE_D(1) + h_t
$$

이것이 가장 중요한 연결고리입니다 — depth transformer의 0번째 입력이 임의의 학습된 토큰이 아니라, 방금 spatial transformer가 만들어낸 그 위치의 context $h_t$라는 것. 이게 두 transformer를 이어주는 다리입니다.

**전체적으로 보면 — 같은 패턴이 두 단계에 걸쳐 반복되는 구조**

```text
Spatial 단계 (위치 축을 누적, d는 매번 D까지 다 채운 상태):
  u_t = "지금 위치가 t다" + "직전 위치(t-1)의 완성된 합(d=1~D)"

Depth 단계 (위치 t 고정, depth 축을 누적):
  v_td = "지금 depth가 d다" + "같은 위치(t)의 부분합(d'=1~d-1)"
  단, d=1이면 그 부분합 자리에 h_t(=spatial transformer 출력)가 대신 들어감
```

$u_t$는 한 위치가 끝난 뒤 다음 위치로 넘어갈 때 쓰는 입력이고, $v_{td}$는 한 위치 안에서 depth를 하나씩 더 깊게 들어갈 때 쓰는 입력입니다. 두 식이 똑같이 생긴 이유는 둘 다 AR하게 하나씩 예측하려면 직전까지 만든 것의 누적 정보를 넣어줘야 한다는 같은 원리를 따르기 때문이고, 차이는 그 직전까지가 가리키는 축(공간 vs depth)과 그로 인한 경계 처리(시작 토큰 vs $h_t$ 재사용)에서 나옵니다.

---

## 14. Spatial Transformer와 Depth Transformer의 연결고리

### 질문: Spatial transformer와 depth transformer의 연결고리는 뭐야?

**연결 지점: $h_t \to v_{t1}$**

가장 직접적인 연결은 다음입니다.

$$
v_{t1} = PE_D(1) + h_t
$$

spatial transformer의 출력 $h_t$가 depth transformer의 첫 입력 $v_{t1}$ 안으로 그대로 들어갑니다. 이게 두 모듈을 잇는 유일하면서도 결정적인 다리입니다.

**왜 이 한 점만으로 연결이 충분한가**

depth transformer가 위치 $t$에서 예측해야 하는 것은 $D$개 코드 $(S_{t1},\dots,S_{tD})$ 전체입니다. 식(9)를 보면 이 모든 코드는 공통적으로 $S_{<t,d}$(다른 위치들 전체)를 조건으로 깔고 있었습니다. depth transformer가 $D$번 코드를 예측하는 동안 다른 위치들에 무슨 정보가 있었는지는 매번 다시 봐야 하는 게 아니라 한 번만 주입되면 충분합니다 — 그것이 $h_t$의 역할입니다.

그래서 depth transformer 내부($d=1,\dots,D$)는 다른 위치 정보를 또 보지 않고, 오직 (1) $h_t$로부터 물려받은 이전 위치들 요약 + (2) 같은 위치 안에서 이미 정해진 더 낮은 depth들($S_{t,<d}$)만 보면서 진행합니다. 이게 식(12)가

$$
v_{t1} = PE_D(1) + h_t
\qquad
v_{td} = PE_D(d) + \sum_{d'=1}^{d-1} e(S_{t,d'}) \quad (d > 1)
$$

으로 나뉘는 이유입니다 — $d=1$일 때만 외부(spatial)에서 온 정보가 필요하고, $d \ge 2$부터는 내부(같은 위치)에서 누적된 정보만 더해주면 됩니다.

**흐름을 한 장으로**

```text
                         [ Spatial Transformer ]
u_1, ..., u_T   ──────►   causal self-attn over T   ──────►   h_1, ..., h_T
                                                                    |
                                          위치 t의 h_t만 뽑아서 ────┘
                                                                    v
                         [ Depth Transformer  (위치 t 전용) ]
              v_t1 = h_t + PE_D(1)
              v_t2 = PE_D(2) + e(S_t1)
              v_t3 = PE_D(3) + e(S_t1) + e(S_t2)
                  :
                                  ──────►   p_t1, p_t2, ..., p_tD
                                            (그 위치의 D개 코드 분포)
```

**한 문장으로**

Spatial transformer는 위치들 사이의 관계를 풀어서 위치별 context $h_t$를 만들고, depth transformer는 그 $h_t$를 시작점으로 삼아 한 위치 안에서 depth들 사이의 관계를 풉니다 — $h_t$가 두 단계의 유일한 인터페이스이고, 이를 통해 다른 위치 정보가 한 번만 압축돼서 depth transformer로 넘어가는 구조입니다. 두 transformer가 동시에 attention을 주고받는 것이 아니라, spatial → ($h_t$) → depth라는 한 방향의 파이프라인으로 연결되어 있습니다.

---

## 15. RQ-Transformer의 NLL Loss — 식(14)

### 질문: RQ-transformer의 NLL loss에 대해서 설명해줘. 식(14)

식(14):

$$
\mathcal{L}_{AR} = \mathbb{E}_{S} \mathbb{E}_{t,d} \left[ -\log p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right) \right]
$$

**NLL이 뭔지부터 — 확률을 손실로 바꾸는 표준적인 방법**

depth transformer가 위치 $t$, depth $d$에서 만들어내는 출력은 식(13)에서 본 $p_{td}$ — codebook의 $K$개 코드에 대한 확률분포(softmax)입니다. 학습의 목표는 실제 정답 코드 $S_{td}$에 높은 확률을 주도록 만드는 것인데, 이를 손실(낮을수록 좋은 값)로 바꾸려면 확률에 $-\log$를 씌웁니다.

$$
\text{NLL} = -\log p\left(S_{td} \mid \text{조건들}\right)
$$

$p$가 1(확실히 맞춤)에 가까우면 $-\log p \to 0$, $p$가 0(완전히 틀림)에 가까우면 $-\log p \to \infty$ — 정답을 맞출 확률을 최대화하는 것과 이 NLL을 최소화하는 것이 같은 목표입니다(분류 문제에서 흔히 쓰는 cross-entropy loss와 동일한 형태).

**조건 부분 — 식(9)가 그대로 다시 등장**

$$
p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right)
$$

이는 식(9)에서 본 그 조건부 분포입니다. 이 값을 실제로 계산하는 과정이 지금까지 본 전체 파이프라인입니다.

$$
S_{<t,d} \xrightarrow{\text{spatial transformer}} h_t \xrightarrow{\text{depth transformer (} S_{t,<d} \text{와 함께)}} p_{td} = p\left(\cdot \mid S_{<t,d}, S_{t,<d}\right)
$$

식(13)의 $p_{td}$가 바로 식(14)의 $p(S_{td}\mid\cdots)$입니다 — 모델이 출력한 분포에서, 실제 정답 코드 $S_{td}$가 받은 확률값만 뽑아서 $-\log$를 취하는 것입니다.

**두 개의 기댓값이 의미하는 것**

- $\mathbb{E}_{t,d}$: 하나의 이미지 안에서, $T \times D$개의 모든 (위치, depth) 쌍에 대해 NLL을 구하고 평균을 낸다는 뜻 ($T=4, D=3$ 예시라면 12개의 NLL 값을 평균).
- $\mathbb{E}_{S}$: 이 이미지 하나당 평균 NLL을, 학습 데이터셋 전체의 이미지들($S$는 한 이미지의 코드맵 전체)에 대해 또 평균을 낸다는 뜻.

실제 구현에서는 기댓값을 정확히 계산할 수 없으므로 미니배치 평균으로 근사합니다 — 배치에 들어온 이미지들 각각에 대해 그 이미지의 $T \times D$개 코드 전체의 NLL을 (병렬로 한 번에) 계산하고, 배치와 $(t,d)$ 전체에 대해 평균낸 값을 loss로 backprop합니다.

**학습 시 중요한 디테일 — Teacher Forcing**

식(14)에서 조건 $S_{<t,d}, S_{t,<d}$는 모델이 실제로 예측한 값이 아니라 RQ-VAE가 진짜 이미지를 인코딩해서 만든 정답 코드입니다. 학습 시에는 매 위치·매 depth에서 모델이 직전에 뭘 예측했든 상관없이 정답을 그대로 다음 입력으로 넣어줍니다 — 이것이 teacher forcing입니다. 그래서 학습 단계에서는 $T \times D$개 코드 전체를 한 번의 forward pass로 병렬 계산할 수 있습니다(causal mask 덕분에, 정답이 이미 다 주어져 있으니 순차적으로 하나씩 생성할 필요가 없습니다).

(이것이 바로 exposure bias가 생기는 지점입니다 — 학습할 때는 항상 정답을 봤는데, 추론할 때는 모델 자신이 만든 예측을 다음 입력으로 써야 하니 discrepancy가 생기고, 이를 줄이려고 soft labeling/stochastic sampling을 도입합니다.)

**정리**

| 구성 요소 | 의미 |
| --- | --- |
| $p(S_{td} \mid S_{<t,d}, S_{t,<d})$ | spatial+depth transformer 파이프라인이 만든 예측 분포에서, 정답 코드가 받은 확률 |
| $-\log(\cdot)$ | 확률 → 손실로 변환 (cross-entropy와 동일 형태) |
| $\mathbb{E}_{t,d}$ | 한 이미지 안의 $T \times D$개 코드 전체에 대한 평균 |
| $\mathbb{E}_{S}$ | 데이터셋 전체 이미지에 대한 평균 |
| 학습 방식 | teacher forcing — 정답 코드를 조건으로 넣고 한 번에 병렬 계산 |

**한 줄 요약**: 식(14)는 이미지 하나에 들어있는 $T \times D$개의 코드 각각을, 식(9)가 정의한 조건(spatial+depth transformer가 만들어낸 분포)으로 얼마나 잘 맞추는지를 cross-entropy로 측정하고, 그것을 모든 코드·모든 이미지에 대해 평균낸 것이 전체 학습 손실입니다.

---

## 16. Exposure Bias

### 질문: 3.2.3 이해하려고. Exposure bias가 뭐야?

Exposure bias는 학습할 때 모델이 보는 입력과 추론할 때 모델이 실제로 보게 되는 입력이 서로 다르다는 데서 생기는 문제입니다. 바로 앞의 teacher forcing이 원인입니다.

**학습 때와 추론 때, 모델이 보는 게 다르다**

- 학습 시(teacher forcing): 식(14)에서 본 대로, $S_{td}$를 예측할 때 조건 $S_{<t,d}$와 $S_{t,<d}$는 항상 RQ-VAE가 만든 정답 코드 그 자체입니다. 모델이 직전 스텝에서 무엇을 예측했든 상관없이, 다음 입력으로는 항상 진짜 정답이 들어갑니다.
- 추론 시: 정답이 없습니다. 모델이 위치 1의 코드를 직접 샘플링하고, 그걸 다음 입력으로 써서 위치 2를 예측하고, 또 그 예측을 입력으로 써서 위치 3을 예측하는 식으로 — 자기 자신이 만든 (틀릴 수도 있는) 예측을 계속 다음 입력으로 재사용합니다.

**왜 이게 문제가 되는가**

학습 중에 모델은 항상 완벽한 과거만 보고 다음을 예측하는 법을 배웁니다. 추론할 때는 한 번이라도 틀린 코드를 뽑으면 그 틀린 코드가 다음 예측의 입력으로 들어갑니다. 모델은 이런 약간 틀린 입력을 학습 중에 한 번도 본 적이 없으니, 그 다음 예측은 더 헷갈리고, 그 결과 또 틀릴 가능성이 높아지고 — 이게 누적됩니다.

> "the exposure bias is known to deteriorate the performance of an AR model due to the error accumulation from the discrepancy of predictions in training and inference"

학습-추론 간의 입력 분포 불일치 → 한 번의 오차가 다음 오차를 유발 → 시퀀스가 길어질수록 오차가 누적되는 것이 exposure bias의 핵심입니다. exposure라는 이름은 모델이 학습 중에 노출(expose)된 입력 분포가 실제 추론 때 노출되는 입력 분포와 다르다는 뜻에서 왔습니다.

**RQ-Transformer에서는 이게 두 배로 일어남**

일반 AR 모델은 시퀀스 길이($t$) 방향으로만 오차가 누적되지만, RQ-Transformer는 depth 방향으로도 누적됩니다.

> "the prediction errors can also accumulate along with the depth D, since finer estimation of the feature vector becomes harder as d increases"

depth $d$가 깊어질수록 그 시점에 양자화해야 하는 대상은 점점 작아지는 잔차입니다. depth 1, 2에서 조금이라도 잘못된 코드를 뽑았다면, depth 3, 4가 양자화해야 할 잔차 자체가 진짜와는 다른 값이 되어버립니다 — depth가 깊어질수록 미세한 디테일을 맞추는 게 원래도 어려운데(finer estimation), 거기에 앞선 depth의 오차까지 얹혀서 더 어려워집니다.

**Scheduled sampling을 쓰지 않은 이유**

학습-추론 discrepancy를 줄이는 표준적인 방법으로 scheduled sampling이 있었지만, 논문은 이를 쓰지 않은 이유를 밝힙니다.

> "Scheduled sampling is a way to reduce the discrepancy. However, it is unsuitable for a large-scale AR model, since multiple inferences are required at each training step and increase the training cost."

Scheduled sampling은 학습 중에도 가끔 모델 자신의 예측을 입력으로 써보게 하는 방법인데, 그러려면 학습 스텝마다 모델로 여러 번 추론을 돌려야 해서(teacher forcing의 한 번의 forward pass로 전체 병렬 계산이라는 효율성이 깨짐) 대규모 모델에는 너무 비쌉니다.

그래서 대신 RQ-VAE의 codebook 임베딩들 사이의 기하학적 관계를 활용하는, 추가 추론이 필요 없는 방법 — soft labeling과 stochastic sampling을 제안합니다.

**정리**

| | 학습 (teacher forcing) | 추론 |
| --- | --- | --- |
| 다음 입력으로 쓰는 값 | 항상 정답 코드 | 모델이 직접 샘플링한 (틀릴 수 있는) 코드 |
| 오차 발생 여부 | 없음 (정답만 보니까) | 한 번 틀리면 그 틀린 값이 계속 전파됨 |
| RQ-Transformer 특이점 | — | 위치($t$) 방향 누적 + depth($d$) 방향 누적까지 이중으로 발생 |

**한 줄 요약**: exposure bias는 학습 때는 항상 정답을 보고 배우지만 추론 때는 자기 자신의 예측을 보고 다음을 만들어야 한다는 근본적인 불일치에서 오는 오차 누적 문제이고, RQ-Transformer에서는 이것이 위치 축과 depth 축 양쪽에서 동시에 일어나기 때문에 특히 신경써서 다뤄야 했던 것입니다.

---

## 17. Exposure Bias가 GPT 등 다른 AR 모델에도 공통적인가

### 질문: 그럼 이게 teacher-forcing으로 학습하는 계열, GPT 등의 모델에 다 공통적으로 발생하는 현상이야?

네, 이것은 teacher forcing으로 학습하는 모든 autoregressive 모델에 구조적으로 내재된 문제입니다. GPT 계열 언어모델도 정확히 같은 현상을 겪습니다.

**원조는 NLP — Bengio et al. 2015**

논문이 인용한 reference [2](scheduled sampling 논문)가 이 문제를 처음 명확히 정리한 논문입니다. Exposure bias라는 용어 자체가 NLP의 sequence generation(기계번역, 텍스트 요약) 연구에서 나왔습니다. GPT가 등장하기 한참 전부터 RNN 기반 seq2seq 모델에서 이미 인지되던 문제이고, GPT 같은 transformer 기반 모델도 학습 방식(teacher forcing)이 같으니 같은 문제를 그대로 물려받습니다.

**GPT에서 정확히 어떻게 나타나는가**

- GPT 학습 시: "The cat sat on the ___"을 학습할 때, 모델이 "mat"이 아니라 "dog"를 예측했더라도, 다음 토큰 예측의 입력으로는 실제 정답인 "mat"이 그대로 들어갑니다 — 모델의 틀린 예측은 loss에는 반영되지만 다음 입력에는 영향을 주지 않습니다.
- GPT 추론(생성) 시: 모델이 "dog"라고 잘못 생성했다면, 다음 토큰을 예측할 때 입력은 "The cat sat on the dog ___"가 됩니다. 모델은 이런 문맥을 학습 중 한 번도 본 적이 없으니 그 다음 토큰은 더 엉뚱해지기 쉽고, 이것이 누적되면 생성이 갈수록 일관성을 잃거나 반복에 빠지는 현상으로 나타납니다.

**왜 GPT는 이걸 RQ-Transformer만큼 적극적으로 다루지 않았나**

1. 스케일이 문제를 어느 정도 덮어줍니다. 모델 크기와 데이터가 충분히 크면 각 스텝의 예측 정확도가 매우 높아져서 한 번 틀릴 확률이 작아지고, 오차 누적의 실질적 영향이 줄어듭니다.
2. Vocabulary의 성격이 다릅니다. 언어모델은 텍스트라는 중복성(redundancy)이 높은 신호를 다뤄서 한 단어를 틀려도 문맥이 의미를 복구해줄 여지가 있습니다. 반면 이미지 코드는 depth가 깊어질수록 점점 미세한 잔차를 양자화하는 것이라서 한 번 틀리면 그 다음 depth가 양자화해야 할 대상 자체가 어긋나버려 복구할 여지가 적습니다.
3. 대안들의 비용 구조가 다릅니다. scheduled sampling류 방법이 학습 스텝마다 추가 추론을 요구해서 비싸다는 점은 GPT 학습에도 동일하게 적용되는 제약이라, 대규모 언어모델도 이 방법을 잘 쓰지 않습니다. 대신 다른 완화책들이 발전했습니다.

**언어모델 쪽에서 쓰는 비슷한 완화책들**

- Label smoothing — one-hot 타깃 대신 약간 부드러운(soft) 분포를 타깃으로 쓰는 것. RQ-Transformer의 soft labeling과 사상이 거의 동일(다만 RQ-Transformer는 codebook 임베딩 간의 기하학적 거리를 이용해 의미적으로 가까운 코드에 확률을 더 주는, label smoothing의 구조화된 버전).
- RLHF/RL 기반 fine-tuning — 모델 자신이 생성한 시퀀스에 대해 보상을 학습시키는 방식으로, 결국 자기 예측을 입력으로 써보는 경험을 학습 루프에 강제로 끼워넣는 것 — scheduled sampling의 현대적 변형.
- Beam search / sampling 전략 개선 — 추론 시점에서 누적 오차를 줄이려는 접근.

**정리**

| | 공통점 |
| --- | --- |
| 원인 | teacher forcing — 학습 입력(정답)과 추론 입력(자기 예측)의 불일치 |
| 영향받는 모델 | GPT 계열을 포함한 모든 teacher-forcing AR 모델 (RNN, transformer 무관) |
| RQ-Transformer가 특별한 점 | 오차가 위치 축뿐 아니라 depth 축으로도 누적되어 이중으로 심해짐 |

**한 줄 요약**: exposure bias는 teacher forcing 자체의 본질적 부작용이라서 GPT를 포함한 모든 AR 모델에 공통으로 존재하고, RQ-Transformer가 이 절에서 따로 다루는 이유는 이 논문의 구조(depth가 깊어질수록 점점 미세해지는 잔차)가 일반 언어모델보다 오차에 더 취약하기 때문입니다.

---

## 18. Exposure Bias 해결책 — Soft Labeling과 Stochastic Sampling

### 질문: 여기서는 그러면 exposure bias 해결하기 위해 어떤 걸 제안하는 거지?

3.2.3절이 제안하는 것은 두 가지이며, 둘 다 하나의 핵심 도구(코드 임베딩 간 거리 기반 softmax 분포)에서 파생됩니다.

**공통 도구: 거리 기반 categorical 분포 $Q_\tau$**

식(15):

$$
Q_\tau(k \mid z) \propto e^{-\Vert z - e(k) \Vert_2^2 / \tau} \qquad k \in [K]
$$

벡터 $z$가 주어졌을 때, codebook의 모든 코드 $k$에 대해 $z$와 그 코드 임베딩 $e(k)$ 사이의 거리가 가까울수록 더 높은 확률을 주는 분포입니다 — 거리에 $-1$을 곱하고 exponential을 취해서 확률처럼 정규화한 softmax 형태입니다.

핵심은 temperature $\tau$의 역할입니다.

- $\tau \to 0$이면 분포가 점점 뾰족해지다가, 결국 가장 가까운 코드 하나에만 확률 1을 주는 one-hot 분포로 수렴합니다.

$$
Q_0(k \mid z) = \mathbb{1}\left[k = Q(z;C)\right]
$$

즉 원래의 deterministic VQ 연산 $Q(z;C)$와 똑같아집니다.

- $\tau$가 클수록 분포가 평평해져서, 1등 코드뿐 아니라 2등, 3등으로 가까운 코드들에도 어느 정도 확률을 나눠줍니다.

$Q_\tau$는 거리가 가까운 정도에 비례해서 후보 코드들에 확률을 분산시켜놓은, deterministic 양자화의 부드러운(soft) 버전입니다.

**제안 1 — Soft Labeling (학습 타깃을 부드럽게)**

기존 NLL loss(식 14)는 정답 코드 $S_{td}$에 대해 one-hot 타깃으로 학습합니다 — 정답 코드만 100% 맞다고 가르치는 것입니다. 위치 $t$, depth $d$에서의 잔차를 $r_{t,d-1}$이라 할 때, 원래는 $Q_0(\cdot \mid r_{t,d-1})$을 타깃으로 쓰는 셈이었습니다.

Soft labeling은 이 one-hot 타깃 $Q_0$ 대신 부드러운 분포 $Q_\tau(\cdot \mid r_{t,d-1})$을 타깃으로 씁니다. 정답 코드만 정답이 아니라, 정답 코드와 가까운(임베딩 거리가 가까운) 다른 코드들도 어느 정도는 맞는 답으로 인정하게 학습 신호를 바꾸는 것입니다.

왜 exposure bias를 줄이는지: 추론 중에 모델이 정답 코드가 아니라 그와 가까운 다른 코드를 뽑았다고 합시다. one-hot으로 학습된 모델은 정답이 아니면 다 틀린 것이라고만 배워서, 이런 약간 빗나간 입력에 대해 어떻게 대응해야 할지 전혀 학습한 적이 없습니다. 반면 soft labeling으로 학습된 모델은 학습 중에 이미 정답과 가까운 코드들도 비슷한 의미를 가진다는 관계를 배웠기 때문에, 추론 중 약간 빗나간 입력이 들어와도 더 안정적으로 대응할 여지가 생깁니다.

**제안 2 — Stochastic Sampling (코드를 뽑는 방식을 부드럽게)**

이것은 학습할 때 RQ-VAE가 타깃 코드 자체를 어떻게 결정하느냐를 바꿉니다. 원래 식(4)는 deterministic합니다 — 잔차 $r_{d-1}$에 대해 항상 가장 가까운 코드 하나를 $\arg\min$으로 고정해서 선택했습니다.

Stochastic sampling은 이 $\arg\min$(deterministic) 대신, $Q_\tau(\cdot \mid r_{d-1})$ 분포에서 코드를 확률적으로 샘플링합니다. 같은 잔차 $r_{d-1}$이라도 매번 똑같은 코드가 나오는 게 아니라, 가까운 후보들 중에서 (거리에 비례한 확률로) 매번 다른 코드가 뽑힐 수 있습니다.

이 역시 $\tau \to 0$이면 원래의 deterministic 선택과 똑같아지므로, 이는 코드 선택 규칙을 완전히 바꾼 것이 아니라 원래 규칙의 확률적 일반화입니다.

왜 exposure bias에 도움이 되는가: 같은 이미지라도 학습 중에 여러 가지 다른 코드 조합으로 표현될 수 있게 됩니다(논문 표현: "different compositions of codes S for a given feature map"). AR 모델(RQ-Transformer)은 정답 코드 시퀀스는 딱 하나라는 식으로 과적합하지 않고, 약간씩 다른 코드 시퀀스들에도 잘 대응하는 법을 배우게 됩니다 — 이것이 추론 중 모델 자신이 약간 다른(틀린) 코드를 만들어냈을 때 무너지지 않고 견딜 수 있는 robustness로 이어집니다.

**두 방법의 관계 정리**

| | 무엇을 부드럽게 만드는가 | 적용 위치 |
| --- | --- | --- |
| Soft labeling | 학습 손실의 타깃 — 정답을 one-hot 대신 $Q_\tau$로 | RQ-Transformer 학습 시, NLL의 supervision |
| Stochastic sampling | RQ-VAE가 코드를 뽑는 과정 자체 — deterministic argmin 대신 $Q_\tau$에서 샘플링 | RQ-VAE 인코딩 시, 타깃 코드 시퀀스 $S$ 생성 |

둘 다 같은 도구($Q_\tau$)를 쓰지만, 하나는 loss의 정답 분포를 흔들고 다른 하나는 정답 시퀀스 자체를 흔든다는 점이 다릅니다. 둘 다 scheduled sampling처럼 학습 중 추가 추론을 요구하지 않습니다 — $Q_\tau$는 RQ-VAE가 이미 갖고 있는 codebook 거리 정보만으로 계산되는 것이라, 학습 비용을 거의 늘리지 않으면서 exposure bias를 완화하는 것이 이 두 제안의 실용적 강점입니다.

---

## 부록: 핵심 수식 모음

### Residual Quantization (식 3, 4)

$$
\mathrm{RQ}(z; C, D) = (k_1, \dots, k_D) \in [K]^D
$$

$$
r_0 = z
\qquad
k_d = Q(r_{d-1}; C)
\qquad
r_d = r_{d-1} - e(k_d)
$$

### RQ-VAE 코드맵과 부분합 (식 5)

$$
M_{hw} = \mathrm{RQ}\left(E(X)_{hw}; C, D\right)
$$

$$
\hat{Z}^{(d)}_{hw} = \sum_{d'=1}^{d} e\left(M_{hwd'}\right)
$$

### RQ-VAE Loss (식 6, 7)

$$
\mathcal{L}_{recon} = \Vert X - \hat{X} \Vert_2^2
$$

$$
\mathcal{L}_{commit} = \sum_{d=1}^{D} \Vert Z - sg[\hat{Z}^{(d)}] \Vert_2^2
$$

### 코드맵 reshape과 AR factorization (식 8, 9)

$$
\mathbf{S}_t = (S_{t1}, \dots, S_{tD}) \in [K]^D
$$

$$
p(\mathbf{S}) = \prod_{t=1}^{T} \prod_{d=1}^{D} p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right)
$$

### Spatial / Depth Transformer 입력 (식 10, 12)

$$
u_t = PE_T(t) + \sum_{d=1}^{D} e(S_{t-1,d}) \qquad (t > 1)
$$

$$
v_{td} = PE_D(d) + \sum_{d'=1}^{d-1} e(S_{t,d'}) \qquad (d > 1)
$$

$$
v_{t1} = PE_D(1) + h_t
$$

### Spatial / Depth Transformer 출력 (식 11, 13)

$$
h_t = \mathrm{SpatialTransformer}(u_1, \dots, u_t)
$$

$$
p_{td} = \mathrm{DepthTransformer}(v_{t1}, \dots, v_{td})
$$

### NLL Loss (식 14)

$$
\mathcal{L}_{AR} = \mathbb{E}_{S} \mathbb{E}_{t,d}\left[-\log p\left(S_{td} \mid S_{<t,d}, S_{t,<d}\right)\right]
$$

### 거리 기반 categorical 분포 (식 15)

$$
Q_\tau(k \mid z) \propto e^{-\Vert z - e(k) \Vert_2^2 / \tau} \qquad k \in [K]
$$

### 계산 복잡도 비교

$$
\text{Naive: } O\left(N T^2 D^2\right)
\qquad
\text{RQ-Transformer: } O\left(N_{spatial} T^2 + N_{depth} T D^2\right)
$$
