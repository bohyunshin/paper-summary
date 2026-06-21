# VQ-VAE (Neural Discrete Representation Learning) 스터디 노트

> van den Oord, Vinyals, Kavukcuoglu (DeepMind), NeurIPS 2017
> 정리: Billy / Claude 대화 기반 (3.1 ~ 3.3절 + 식 3 전개)

---

## 목차

1. [3.1 Discrete Latent Variables](#1-31-discrete-latent-variables)
2. [VAE vs AE 차이](#2-vae-vs-ae-차이)
3. [Posterior $q(z\mid x)$ vs Prior $p(z)$](#3-posterior-qzmidx-vs-prior-pz)
4. [$p(z\mid x)$에서 $q(z\mid x)$로: Variational Inference의 필요성](#4-pzmidx에서-qzmidx로-variational-inference의-필요성)
5. [VAE에서 ELBO로 이어지는 전체 흐름](#5-vae에서-elbo로-이어지는-전체-흐름)
6. [ELBO와 KL$(q\Vert p(z\mid x))$의 관계 (4단계 상세)](#6-elbo와-klq-vert-pzmidx의-관계-4단계-상세)
7. [왜 discrete representation을 택했는가](#7-왜-discrete-representation을-택했는가)
8. [Discrete vs Continuous 표현의 구체적 예시](#8-discrete-vs-continuous-표현의-구체적-예시)
9. [KL divergence가 상수 $\log K$가 되는 이유 (직접 계산)](#9-kl-divergence가-상수-log-k가-되는-이유-직접-계산)
10. [식 2: $z_q(x)$가 디코더의 입력](#10-식-2-zqx가-디코더의-입력)
11. [Straight-Through Estimator](#11-straight-through-estimator)
12. [식 3: 전체 Loss 함수](#12-식-3-전체-loss-함수)
13. [재구성 항이 어떻게 Loss가 되는가](#13-재구성-항이-어떻게-loss가-되는가)
14. [VQ Loss 항 상세 분석](#14-vq-loss-항-상세-분석)
15. [Stop-gradient의 위치가 갖는 의미](#15-stop-gradient의-위치가-갖는-의미)
16. [VQ Loss / Commitment Loss가 왜 필요한가 (근본 이유)](#16-vq-loss--commitment-loss가-왜-필요한가-근본-이유)
17. [구체적 분포 가정 하에서 ELBO → 식 3 전체 유도](#17-구체적-분포-가정-하에서-elbo--식-3-전체-유도)
18. [각 파라미터에 대한 Gradient 유도](#18-각-파라미터에-대한-gradient-유도)
19. [$N$개 latent로의 확장과 $\log p(x)$ 평가](#19-n개-latent로의-확장과-log-px-평가)
20. [3.3 Prior](#20-33-prior)

---

## 1. 3.1 Discrete Latent Variables

### 모델 구성 요소

VQ-VAE는 세 부분으로 구성된다.

1. **인코더**: 입력 $x$를 받아 연속 벡터 $z_e(x)$를 출력
2. **임베딩 공간(codebook)** $e \in \mathbb{R}^{K \times D}$: $K$개의 벡터 $e_1, \dots, e_K$가 들어있는 테이블. $K$는 discrete latent space의 크기, $D$는 각 임베딩 벡터의 차원
3. **디코더**: codebook 벡터를 받아 $x$를 복원

### 식 1 — Posterior $q(z\mid x)$

$$
q(z=k \mid x) =
\begin{cases}
1 & \text{if } k = \arg\min_j \lVert z_e(x) - e_j \rVert_2 \\
0 & \text{otherwise}
\end{cases}
$$

인코더 출력 $z_e(x)$를 codebook의 $K$개 벡터와 비교해서, **가장 가까운(L2 거리 최소) 벡터의 인덱스 $k$**를 결정론적으로 선택한다. Soft한 확률 분포가 아니라 one-hot — 즉 deterministic한 nearest-neighbor lookup.

### 식 2 — Decoder 입력

$$
\begin{aligned}
z_q(x) &= e_k \\
k &= \arg\min_j \lVert z_e(x) - e_j \rVert_2
\end{aligned}
$$

선택된 인덱스 $k$에 해당하는 codebook 벡터 $e_k$ 자체가 디코더의 입력이 된다. 인코더 출력을 codebook에서 가장 가까운 점으로 "스냅(snap)"시키는 것이 discretization bottleneck.

### Figure 1 (우측 그림) 직관

연속 공간에 점들이 흩어져 있고, $z_e(x)$가 그중 $e_2$에 가장 가깝다고 표시됨. 이것이 "양자화(quantization)"의 의미 — 연속값을 미리 정해둔 유한 개의 대표값 중 하나로 강제로 매핑하는 것. 신호 처리의 quantization(아날로그 신호 → 디지털 레벨)과 동일한 개념을 latent space에 적용.

---

## 2. VAE vs AE 차이

### Autoencoder (AE)

목표: $x$를 압축했다가 다시 복원.

$$
x \to \text{Encoder} \to z (\text{하나의 고정된 벡터}) \to \text{Decoder} \to \hat{x}
$$

$$
\mathcal{L} = \lVert x - \hat{x} \rVert^2
$$

$z$는 **결정론적인 하나의 점**. 같은 $x$를 넣으면 항상 같은 $z$. $z$ space에 분포 구조를 강제하지 않아 듬성듬성하고 구멍이 많은 공간이 됨.

### Variational Autoencoder (VAE)

목표: 생성 모델 — $z$를 샘플링하면 그럴듯한 새 $x$를 만들어낼 수 있어야 함.

1. 인코더가 분포의 파라미터 $(\mu, \sigma^2)$를 출력 → $q(z\mid x) = \mathcal{N}(\mu, \sigma^2)$
2. $z$ space 전체가 prior $p(z) = \mathcal{N}(0, I)$를 따르도록 강제

$$
\mathcal{L} = \text{재구성 오차} + \mathrm{KL}\big(q(z\mid x) \mathrel{\Vert} p(z)\big)
$$

KL term이 $q(z\mid x)$를 표준정규분포에 가깝게 끌어당겨 $z$ space를 매끄럽게 만듦 → 생성 가능.

### 비교표

| | AE | VAE |
|---|---|---|
| 인코더 출력 | 고정된 벡터 $z$ | 분포의 파라미터 $(\mu, \sigma)$ |
| $z$를 얻는 방법 | 그냥 그 벡터 | 샘플링 (reparameterization trick) |
| Loss | 재구성 오차만 | 재구성 오차 + KL |
| $z$ space 구조 | 보장 없음 (구멍 있음) | 매끄럽고 연속적 |
| 새 샘플 생성 | 잘 안 됨 | 가능 |

### VQ-VAE의 위치

- AE처럼: 인코더 출력 $z_e(x)$가 deterministic
- VAE처럼: "discrete VAE" 프레임 안에서 정당화되고, 학습 후 별도로 prior $p(z)$를 PixelCNN/WaveNet으로 학습해서 생성 가능
- 단, KL term은 (uniform prior + one-hot posterior 조합 때문에) 상수가 되어 그래디언트에 기여 안 함 → 학습은 사실상 AE처럼 "재구성 + 보조항"으로 진행되고, 생성 능력은 별도의 **2단계 과정**으로 얻음

---

## 3. Posterior $q(z\mid x)$ vs Prior $p(z)$

### $p(z)$ — Prior

데이터 $x$를 보기 전에 $z$가 어떤 모양이어야 한다고 미리 정해놓은 분포. 보통 $\mathcal{N}(0, I)$.

- $x$에 의존하지 않음
- 학습 가능한 파라미터 없음 (고정)
- 생성 시 사용: $p(z)$에서 $z$를 샘플링해 디코더에 입력

### $q(z\mid x)$ — Posterior (인코더가 만드는 분포)

특정 $x$가 주어졌을 때 그에 대응하는 $z$가 어디 있을지 추정하는 분포.

- $x$에 의존 (인코더가 $\mu(x), \sigma(x)^2$를 계산)
- 인코더 네트워크 파라미터로 학습됨
- 인코딩 시 사용

### 왜 둘 다 필요한가

진짜 사후분포 $p(z\mid x)$는 계산 불가능 (분모 $p(x) = \int p(x\mid z)p(z)\mathrm{d}z$가 적분 불가능). 그래서 $q(z\mid x)$라는 **근사 분포**를 인코더로 학습시켜 $p(z\mid x)$를 대신함 — 이것이 "variational"의 의미.

### KL term이 두 분포를 묶는 지점

$$
\mathrm{KL}\big(q(z\mid x) \mathrel{\Vert} p(z)\big)
$$

- $q(z\mid x)$가 각 $x$마다 제멋대로 흩어지면 $z$ space가 구멍 나고 불연속적이 됨
- $q(z\mid x)$를 $p(z) = \mathcal{N}(0,I)$에 가깝게 끌어당김
- 결과: 모든 $x$의 $q(z\mid x)$들이 $z$ space 전체에 매끄럽게 겹쳐 분포

### VQ-VAE에서 달라지는 지점

$q(z\mid x)$ = deterministic one-hot, $p(z)$ = uniform categorical → $\mathrm{KL} = \log K$ (상수) → "두 분포를 가깝게 만드는" 압박이 학습 중 전혀 작동하지 않음. 그래서 학습 후 $p(z)$를 PixelCNN/WaveNet으로 **별도로 다시 학습**(3.3절).

---

## 4. $p(z\mid x)$에서 $q(z\mid x)$로: Variational Inference의 필요성

### 베이즈 정리로 본 "진짜" 사후분포

원래 알고 싶은 것은 $p(z\mid x)$:

$$
p(z\mid x) = \frac{p(x\mid z)p(z)}{p(x)}
$$

분자는 계산 가능(디코더 $p(x\mid z)$, prior $p(z)$). 문제는 분모:

$$
p(x) = \int p(x\mid z)p(z)\mathrm{d}z
$$

고차원 연속 공간에서 이 적분은 닫힌 형태로 풀 수 없음 → $p(z\mid x)$ 직접 계산 불가능.

### 인코더의 진짜 역할

이상적으로 인코더가 출력해야 하는 것은 $p(z\mid x)$. 계산 불가능하므로 대신 학습 가능한 분포 $q(z\mid x)$(보통 정규분포)로 **근사**시킴. 이것이 variational inference — 어려운 분포를 다루기 쉬운 분포족으로 근사하고, 그 근사 품질을 KL divergence로 측정:

$$
\mathrm{KL}\big(q(z\mid x) \mathrel{\Vert} p(z\mid x)\big)
$$

이걸 최소화하면 $q$가 진짜 $p(z\mid x)$에 가까워짐. 그런데 이 KL 식 안에도 $p(z\mid x)$(분모에 $p(x)$ 숨어있음)가 들어있어 여전히 직접 계산 불가능.

### ELBO의 등장

대수 조작을 통해:

$$
\log p(x) = \mathrm{ELBO} + \mathrm{KL}\big(q(z\mid x) \mathrel{\Vert} p(z\mid x)\big)
$$

KL은 항상 0 이상이므로 ELBO는 $\log p(x)$의 하한. 핵심: **ELBO는 $p(z\mid x)$ 없이 계산 가능**($q(z\mid x), p(z), p(x\mid z)$만으로 표현됨). 그래서 직접 계산 불가능한 KL을 최소화하는 대신 계산 가능한 ELBO를 최대화하는 쪽으로 문제를 치환.

### 한 줄 요약

- 원하는 것: $p(z\mid x)$ (진짜 사후분포)
- 문제: $p(x)$의 적분 불가능
- 해법: $q(z\mid x)$로 근사
- 학습 방법: KL을 직접 줄일 수 없으니 수학적으로 동치인 ELBO를 최대화

---

## 5. VAE에서 ELBO로 이어지는 전체 흐름

### 출발점

$\log p(x)$를 최대화하고 싶지만 $p(x) = \int p(x\mid z)p(z)\mathrm{d}z$는 적분 불가능.

### 1단계: $q(z\mid x)$를 끼워넣기

$$
\log p(x) = \log \int p(x,z)\mathrm{d}z = \log \int q(z\mid x) \cdot \frac{p(x,z)}{q(z\mid x)}\mathrm{d}z = \log \mathbb{E}_{q(z\mid x)}\left[\frac{p(x,z)}{q(z\mid x)}\right]
$$

### 2단계: Jensen's inequality 적용

$\log$는 concave 함수이므로 $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$:

$$
\log p(x) \ge \mathbb{E}_{q(z\mid x)}\left[\log \frac{p(x,z)}{q(z\mid x)}\right] =: \mathrm{ELBO}
$$

"Evidence"는 $p(x)$를 가리키는 통계학 용어 — ELBO는 그 evidence의 하한(lower bound).

### 3단계: ELBO를 두 항으로 분해

$p(x,z) = p(x\mid z)p(z)$이므로:

$$
\begin{aligned}
\mathrm{ELBO} &= \mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z) + \log p(z) - \log q(z\mid x)\big] \\
&= \mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big] - \mathbb{E}_{q(z\mid x)}\big[\log q(z\mid x) - \log p(z)\big] \\
&= \underbrace{\mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big]}_{\text{재구성 항}} - \underbrace{\mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z)\big)}_{\text{KL term}}
\end{aligned}
$$

이것이 VAE loss의 정확한 형태.

### 4단계: ELBO와 KL$(q\Vert p(z\mid x))$의 관계

(상세 전개는 [6번 섹션](#6-elbo와-klq-vert-pzmidx의-관계-4단계-상세) 참고)

$$
\log p(x) = \mathrm{ELBO} + \mathrm{KL}\big(q(z\mid x) \mathrel{\Vert} p(z\mid x)\big)
$$

ELBO를 올리면 $\log p(x)$가 고정값이므로 자동으로 KL이 줄어듦 → $q(z\mid x)$가 $p(z\mid x)$에 가까워지는 효과.

### 5단계: Reparameterization trick

$\mathbb{E}_{q(z\mid x)}[\log p(x\mid z)]$를 계산하려면 $z \sim q(z\mid x)$ 샘플링이 필요한데, 샘플링은 미분이 안 됨(확률적 연산이라 backprop 그래프가 끊김).

$q(z\mid x) = \mathcal{N}(\mu(x), \sigma(x)^2)$일 때:

$$
z = \mu(x) + \sigma(x) \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0,1)
$$

확률성을 $\epsilon$(파라미터와 무관한 노이즈)으로 빼내면 $\mu(x), \sigma(x)$는 결정론적 함수가 되어 backprop 가능.

### 요약 도식

```
log p(x) [계산 불가]
   → Jensen's inequality → ELBO [계산 가능한 하한]
   → ELBO 분해 → 재구성 항 + KL 정규화 항
   → reparameterization trick → SGD로 학습 가능
```

### VQ-VAE가 이 흐름에서 깨뜨리는 두 지점

1. $q(z\mid x)$가 정규분포가 아니라 **deterministic one-hot** (식 1) → reparameterization trick 자체가 불필요, 대신 식 2의 argmin이 직접 $z$를 결정
2. KL term이 **상수**가 되어 사라짐 (uniform prior + one-hot posterior)

그 결과 ELBO에서 남는 건 사실상 재구성 항 하나뿐이고, 거기에 VQ 학습을 위한 보조 loss(codebook 업데이트, commitment loss)가 추가되는 것이 식 3의 구조.

---

## 6. ELBO와 KL$(q \Vert p(z\mid x))$의 관계 (4단계 상세)

### 결합분포의 두 가지 분해

$$
p(x,z) = p(x\mid z)p(z) \qquad \text{(디코더 방향)}
$$
$$
p(x,z) = p(z\mid x)p(x) \qquad \text{(진짜 사후분포 방향)}
$$

둘 다 같은 $p(x,z)$를 가리키는 등가식.

### ELBO를 정의에서부터 재전개

$$
\mathrm{ELBO} = \mathbb{E}_q\big[\log p(x,z) - \log q(z\mid x)\big]
$$

두 번째 분해 $p(x,z) = p(z\mid x)p(x)$를 대입:

$$
\begin{aligned}
\mathrm{ELBO} &= \mathbb{E}_q\big[\log p(z\mid x) + \log p(x) - \log q(z\mid x)\big] \\
&= \log p(x) + \mathbb{E}_q\big[\log p(z\mid x) - \log q(z\mid x)\big] \qquad (\log p(x)\text{는 } z\text{에 무관, 기댓값 밖으로}) \\
&= \log p(x) - \mathbb{E}_q\big[\log q(z\mid x) - \log p(z\mid x)\big] \\
&= \log p(x) - \mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z\mid x)\big)
\end{aligned}
$$

### 핵심 등식

$$
\mathrm{ELBO} = \log p(x) - \mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z\mid x)\big)
$$

양변 정리:

$$
\log p(x) = \mathrm{ELBO} + \mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z\mid x)\big)
$$

### 이 식이 말해주는 세 가지

**1) ELBO는 항상 $\log p(x)$ 이하**: KL은 항상 $\ge 0$이므로 $\log p(x) = \mathrm{ELBO} + (\ge 0) \Rightarrow \log p(x) \ge \mathrm{ELBO}$.

**2) ELBO를 올리는 것 = KL을 줄이는 것**: $\log p(x)$는 데이터가 주어지면 고정값 (모델 파라미터와 무관). 좌변이 고정이므로 ELBO가 커지는 만큼 KL이 작아져야 등식 유지.

**3) 계산 가능성이 정반대**: $\mathrm{KL}(q\mathrel{\Vert} p(z\mid x))$는 $p(z\mid x)$가 들어있어 직접 계산 불가능. ELBO는 $p(x\mid z), p(z), q(z\mid x)$만으로 표현되어 계산 가능($p(z\mid x)$가 전혀 안 나타남). 계산 불가능한 KL을 직접 줄이는 문제를, 계산 가능한 ELBO를 올리는 문제로 치환.

### 그림으로 이해

```
        log p(x)        ← 고정된 천장 (데이터가 정해지면 변하지 않음)
           │
       ┌───┴───┐
       │  gap  │  ← KL(q(z|x) || p(z|x))  (q가 진짜 posterior와 얼마나 다른지)
       ├───────┤
       │ ELBO  │  ← 우리가 실제로 최적화하는 양
       └───────┘
```

ELBO를 위로 밀어올리면 천장은 고정이니 자연히 gap(KL)이 줄어듦 — 직접 타겟할 수 없는 것을 간접적으로 압박해서 달성하는 구조.

---

## 7. 왜 Discrete Representation을 택했는가

### 1) Categorical인 것과 deterministic인 것은 별개의 선택

discrete representation을 쓰려면 $q(z\mid x)$가 categorical 분포여야 하는 건 맞지만, categorical이라고 해서 꼭 deterministic이어야 하는 건 아님. 이론적으로는 $q(z=k\mid x) = \text{softmax}(\text{거리 기반 점수})$처럼 확률적인 soft categorical도 가능했음. 실제로 Gumbel-softmax / Concrete distribution이 이 방향.

### 2) Deterministic을 택한 이유 — 분산(variance) 문제

기존 discrete VAE 학습법(NVIL, VIMCO)은 score-function estimator(REINFORCE류)를 써서 그래디언트 분산이 매우 큼. Gumbel-softmax/Concrete는 학습 초반엔 그래디언트 분산이 낮지만 편향(bias)이 있고, 학습 후반엔 분산이 커지지만 편향이 없어지는 trade-off가 있음. VQ-VAE는 샘플링 자체를 없애 이 문제를 원천 회피.

### 3) Posterior collapse 회피

강력한 autoregressive decoder와 짝지을 때 일반 VAE는 latent $z$를 무시하는 posterior collapse가 생김. $q(z\mid x)$가 deterministic이고 KL이 상수로 고정되면 "KL을 줄이려고 latent를 무시하는" 압박 자체가 loss에 존재하지 않음.

### 4) 데이터 자체가 본질적으로 discrete

언어는 본질적으로 discrete, 음성도 일련의 기호, 이미지도 종종 언어로 간결히 설명 가능. 모델링 대상이 본래 discrete 구조를 가지면 discrete latent가 더 자연스러운 표현.

### 5) Reasoning / planning에 적합

복잡한 추론, 계획, predictive learning에 자연스럽게 맞음 (예: "비가 오면 우산을 쓴다" 식의 조건문 추론). discrete 심볼 단위가 연속 벡터보다 이런 논리적 추론에 더 적합.

### 6) 중요한 특징을 압축적으로 잡아냄 — 노이즈 무시

latent space를 효과적으로 사용할 수 있어, 이미지의 객체, 음성의 음소, 텍스트의 메시지처럼 여러 차원에 걸친 중요한 특징을 모델링하고, 노이즈나 인지하기 힘든 local 디테일에는 capacity를 쓰지 않음. 음성 실험에서 화자 음색(local, 저수준)은 사라지고 발화 내용(전역, 고수준)만 latent에 남는 것이 이 현상의 증거.

### 7) 압축 + 실용적 이점

$128\times128\times3$ 이미지를 $32\times32\times1$ ($K=512$)로 압축하면:

$$
\frac{128 \times 128 \times 3 \times 8}{32 \times 32 \times 9} \approx 42.6 \text{배 압축 (비트 기준)}
$$

### 정리표

| 이유 | 무엇을 얻는가 |
|---|---|
| 데이터의 본질이 discrete (언어, 음소) | 표현의 자연스러움 |
| reasoning/planning에 적합 | downstream 활용성 |
| 압축 강제 → 중요한 global feature만 보존 | 노이즈에 강건한 표현 |
| 상수 KL → posterior collapse 없음 | 학습 안정성, latent가 실제로 쓰임 |
| codebook 인덱스 = 압축 표현 | 명시적 압축률, 토큰화 가능 |

> **연결 포인트**: codebook 인덱스를 "토큰"처럼 다루면 시퀀스 모델(PixelCNN, WaveNet, 혹은 Transformer)을 얹을 수 있음 — 이것이 RQ-VAE/TIGER/semantic ID 계열로 이어지는 핵심 다리.

---

## 8. Discrete vs Continuous 표현의 구체적 예시

### 예시 1: 색깔

- **Continuous**: RGB $(0.62, 0.31, 0.78)$ — 무한히 미세한 차이 표현 가능
- **Discrete**: "빨강/파랑/보라/초록" 중 하나 — 비슷한 색은 모두 같은 카테고리로 뭉침 (clustering 효과)

### 예시 2: 이미지 latent (ImageNet 실험)

- **Continuous VAE**: $z = (0.23, -1.04, 0.88, \dots)$, 사진이 미세하게 바뀌면 $z$도 미세하게 변화
- **VQ-VAE**: $32\times32$ 위치마다 "codebook 512개 중 몇 번 패턴에 가장 가까운가"를 정수 인덱스로 그리드화. 미세한 차이는 무시하고 같은 코드로 스냅

### 예시 3: 음성 (논문 4.3절 실험)

같은 문장을 같은 화자가 두 번 말하면 waveform은 미묘하게 다름(억양, 속도, 톤).

- **Continuous latent**: 미세한 차이까지 $z$에 다 반영
- **VQ-VAE discrete latent**: 음향적 디테일을 버리고 "음소 수준 내용"만 코드로 남김. 재구성 시 waveform 자체는 원본과 다르고 prosody도 변하지만 **말하는 내용(텍스트)은 보존**. 화자 ID만 바꿔 디코딩하면 같은 내용을 다른 목소리로 말하게 함(speaker conversion)

### 비유

- **Continuous** = 온도계 (23.4°C, 23.41°C... 무한히 세밀하지만 의미 단위로 나누기 애매)
- **Discrete** = 계절 ("여름/가을/겨울/봄" — 정밀도는 버렸지만 의사결정엔 더 다루기 쉬움)

VQ-VAE가 노리는 것: 데이터의 미세한 연속적 변동(노이즈에 가까운 것)은 버리고, 의미 있게 구분되는 카테고리만 latent로 남겨서 추론/생성/압축을 쉽게 만드는 것.

---

## 9. KL Divergence가 상수 $\log K$가 되는 이유 (직접 계산)

### 원문

> "by defining a simple uniform prior over $z$ we obtain a KL divergence constant and equal to $\log K$."

### KL divergence 정의

$$
\mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z)\big) = \sum_{k=1}^{K} q(z=k\mid x)\log\frac{q(z=k\mid x)}{p(z=k)}
$$

### 두 분포 대입

**$q(z\mid x)$** (식 1, one-hot, $k^\ast$를 선택된 인덱스라 하면):

$$
q(z=k\mid x) =
\begin{cases}
1 & k = k^\ast \\
0 & k \ne k^\ast
\end{cases}
$$

**$p(z)$** (uniform prior):

$$
p(z=k) = \frac{1}{K} \quad \text{(모든 } k\text{에 대해)}
$$

### 합을 직접 전개

$q(z=k\mid x)=0$인 항들은 모두 0이 되어 사라짐. $k=k^\ast$인 항 하나만 남음:

$$
\mathrm{KL} = q(z=k^\ast\mid x)\cdot\log\frac{q(z=k^\ast\mid x)}{p(z=k^\ast)} = 1 \cdot \log\frac{1}{1/K} = \log K
$$

### 왜 "constant"인가

$k^\ast$가 무엇이든(인코더가 어떤 입력 $x$에 대해 어떤 코드를 고르든) 결과는 항상 $\log K$. $q(z=k^\ast\mid x)$는 항상 1, $p(z=k^\ast)$는 균등분포라 $k^\ast$가 몇 번이든 항상 $1/K$.

### 학습 관점에서의 의미

$$
\frac{\partial(\log K)}{\partial(\text{인코더 파라미터})} = 0
$$

상수를 미분하면 0 → **이 KL 항은 학습에 그래디언트를 전혀 기여하지 않음** → 식 3에서 제외 가능 (실제로 식 3에는 reconstruction + VQ loss + commitment loss 세 개뿐, KL 항 없음).

### 비교표

| | 보통 VAE | VQ-VAE |
|---|---|---|
| $q(z\mid x)$ | $x$마다 다른 평균/분산의 정규분포 | $x$와 무관하게 항상 one-hot |
| $\mathrm{KL}(q\mathrel{\Vert} p)$ | $x$에 따라 값 달라짐, 그래디언트 있음 | 항상 $\log K$, 그래디언트 0 |
| 학습에서의 역할 | 정규화 항으로 실제 작동 | 무시 가능한 상수 |

---

## 10. 식 2: $z_q(x)$가 디코더의 입력

### 전체 파이프라인

$$
x \to [\text{인코더}] \to z_e(x) \to [\text{argmin, 최근접 codebook 탐색}] \to z_q(x) \to [\text{디코더}] \to \hat{x}
$$

- **$z_e(x)$**: 인코더의 직접 출력 (연속 벡터, codebook 스냅 전)
- **$z_q(x)$**: codebook에서 최근접 벡터로 스냅된 후의 값 ($=e_k$) — **디코더로 들어가는 입력**

forward computation 동안 가장 가까운 임베딩 $z_q(x)$(식 2)가 디코더로 전달되고, backward pass 동안 그래디언트는 변형 없이 인코더로 전달된다.

### 디코더는 $z_e(x)$를 직접 보지 않음

디코더에 들어가기 전 반드시 $K$개 codebook 벡터 중 하나로 양자화되어야 하고, 그 결과가 $z_q(x)$. 디코더는 항상 codebook을 거친 $z_q(x)$만 입력으로 받음.

---

## 11. Straight-Through Estimator

### 문제: argmin은 미분이 안 된다

$$
z_q(x) = e_k, \quad k = \arg\min_j \lVert z_e(x) - e_j \rVert_2
$$

학습을 위해 인코더 파라미터로 미분(backprop)해야 하는데, 경로 중간의 argmin이 문제:

1. argmin의 출력(정수 인덱스 $k$)은 연속적이지 않아 미분 대상이 안 됨
2. $z_e(x)$를 조금 움직여도 대부분 $k$가 안 바뀜 → 그래디언트가 거의 항상 0
3. $k$가 바뀌는 경계에서는 $z_q(x)$가 $e_k$에서 $e_{k'}$로 갑자기 점프 → 그 지점에서 그래디언트 undefined

즉 $z_e(x) \to z_q(x)$ 경로는 계단 함수처럼 작동 — 평평한 구간 기울기 0, 계단 끝 기울기 undefined.

### 만약 그대로 두면

$$
\frac{\partial \mathcal{L}}{\partial(\text{인코더 파라미터})} = \frac{\partial \mathcal{L}}{\partial z_q(x)} \cdot \underbrace{\frac{\partial z_q(x)}{\partial z_e(x)}}_{\text{거의 항상 0 또는 undefined}} \cdot \frac{\partial z_e(x)}{\partial(\text{인코더 파라미터})}
$$

인코더가 그래디언트를 전혀 못 받음 → 학습 불가능.

### 해법: 그래디언트를 그냥 복사해서 통과

식 2에는 진짜 그래디언트가 정의되어 있지 않지만, straight-through estimator와 유사하게 그래디언트를 근사하며, 디코더 입력 $z_q(x)$에서 인코더 출력 $z_e(x)$로 그래디언트를 그대로 복사한다.

$$
\text{forward}: \quad z_q(x) = e_k \quad \text{(argmin으로 진짜 양자화 수행)}
$$
$$
\text{backward}: \quad \frac{\partial \mathcal{L}}{\partial z_e(x)} := \frac{\partial \mathcal{L}}{\partial z_q(x)} \quad \text{(그냥 통째로 복사, argmin 단계는 무시)}
$$

forward에서는 양자화라는 비선형 연산이 있었던 것처럼 작동하지만, backward에서는 그 연산이 없었던 것(identity function)처럼 그래디언트를 흘려보냄. 이름의 유래: 그래디언트가 양자화 단계를 "그냥 뚫고 지나간다(straight through)".

### 직관적 정당화

$z_e(x)$와 $z_q(x)$는 같은 $D$차원 공간에 있고, $e_k$는 $z_e(x)$와 가장 가까운 점이므로 둘은 값이 비슷함. "$z_q(x)$를 이 방향으로 바꾸면 loss가 줄어든다"는 정보는 "$z_e(x)$를 같은 방향으로 바꿔도 비슷하게 loss가 줄어들 것"이라는 정보와 거의 같다고 볼 수 있음 (완전히 정확한 그래디언트는 아니지만 실용적으로 잘 작동하는 근사).

### 도식

```
Forward:   z_e(x)  --[argmin, 진짜 양자화]-->  z_q(x)  --> 디코더 --> 재구성
Backward:  z_e(x)  <--[그냥 그대로 복사]--  ∇z_q(x)  <-- 디코더 <-- Loss
```

(원조: Bengio et al. 2013, 논문 레퍼런스 [3] — binary/discrete 뉴런을 통과하는 그래디언트 추정의 일반적 기법)

---

## 12. 식 3: 전체 Loss 함수

$$
\begin{aligned}
\mathcal{L} =
&\underbrace{\log p(x\mid z_q(x))}_{\text{항 1: 재구성}} \\
&+ \underbrace{\lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2}_{\text{항 2: VQ loss}} \\
&+ \underbrace{\beta\lVert z_e(x) - \mathrm{sg}[e] \rVert_2^2}_{\text{항 3: commitment loss}}
\end{aligned}
$$

### $\mathrm{sg}[\cdot]$ — stop-gradient 연산자

- forward: identity, $\mathrm{sg}[A] = A$
- backward: $\dfrac{\partial \mathrm{sg}[A]}{\partial A} = 0$

sg로 감싸진 변수는 "이 forward에서 쓰긴 하지만 이걸로 거슬러 올라가서 업데이트하지는 마라"는 뜻 — 해당 변수를 동결(freeze)시키는 스위치.

### 항 1: 재구성 loss — $\log p(x\mid z_q(x))$

ELBO의 재구성 항. 디코더가 $z_q(x)$를 받아 원래 $x$를 얼마나 잘 복원하는지 측정.

**학습시키는 대상**: 디코더 파라미터 + (straight-through 경유) 인코더 파라미터. sg가 없어 $z_q(x)$를 통해 인코더·디코더 모두로 그래디언트가 흐름.

### 항 2: VQ loss (codebook 학습) — $\lVert \mathrm{sg}[z_e(x)] - e \rVert^2$

$$
\mathrm{sg}[z_e(x)] \to \text{인코더 출력 동결 (이 항으로 인코더 업데이트 안 됨)}
$$
$$
e \to \text{동결 안 됨 (이 항이 codebook } e\text{를 학습시킴)}
$$

**의미**: codebook의 벡터 $e$를 인코더 출력 $z_e(x)$ 쪽으로 끌어당김. 인코더는 고정, codebook만 이동.

**필요성**: straight-through estimator로 인해 임베딩 $e_i$는 reconstruction loss로부터 그래디언트를 전혀 받지 못하므로, 임베딩 공간 학습을 위해 가장 단순한 dictionary learning 알고리즘인 Vector Quantisation(VQ)을 사용.

### 항 3: Commitment loss — $\beta\lVert z_e(x) - \mathrm{sg}[e] \rVert^2$

$$
z_e(x) \to \text{동결 안 됨 (이 항이 인코더를 학습시킴)}
$$
$$
\mathrm{sg}[e] \to \text{codebook 동결 (이 항으로 codebook 업데이트 안 됨)}
$$

**의미**: 항 2와 정반대 방향 — 인코더 출력 $z_e(x)$를 codebook 벡터 $e$ 쪽으로 끌어당김. codebook 고정, 인코더만 학습.

**"commitment"인 이유**: embedding space의 volume은 차원이 없어서(dimensionless), embedding이 encoder 파라미터만큼 빠르게 학습되지 않으면 임의로 커질 수 있음. commitment loss는 인코더가 하나의 codebook 벡터에 확실히 전념(commit)하도록 강제해 이 발산을 막음.

$\beta = 0.25$ 사용, $0.1\sim2.0$ 범위에서 결과가 크게 변하지 않아 robust함을 확인.

### 전체 그림: 누가 무엇을 학습시키나

| | 디코더 | 인코더(직접) | 인코더(ST estimator 경유) | codebook $e$ |
|---|---|---|---|---|
| 항1 (재구성) | ✓ | – | ✓ | – |
| 항2 (VQ loss) | – | – | – | ✓ |
| 항3 (commitment) | – | ✓ | – | – |

디코더는 첫 번째 loss term만 최적화하고, 인코더는 첫 번째와 마지막 loss 항을 최적화하며, embedding들은 중간 loss 항으로 최적화된다.

### 한 문장 요약

- **항 1**: "디코더야, $z_q(x)$로 원본을 잘 복원해라" (+인코더도 출력 조정)
- **항 2**: "codebook아, 인코더가 내놓는 값들 쪽으로 따라가라"
- **항 3**: "인코더야, 너무 codebook에서 멀어지지 말고 가까운 곳에 정착해라"

---

## 13. 재구성 항이 어떻게 Loss가 되는가

### 질문: $p(x\mid z_q(x))$는 분포인데 어떻게 loss(스칼라)가 되는가?

디코더 신경망은 $z_q(x)$를 입력받아 분포의 **파라미터**를 출력 (예: 이미지 픽셀이 연속값이면 평균 $\mu$를 출력, $p(x\mid z_q(x)) = \mathcal{N}(x;\mu,\sigma^2 I)$, 분산은 보통 고정).

$x$는 이미 갖고 있는 **실제 관측 데이터**이므로, $\log p(x\mid z_q(x))$에서 $z_q(x)$는 분포의 파라미터($\mu$)를 결정하고 $x$는 이미 정해진 값을 대입 → **결과는 분포가 아니라 하나의 숫자(스칼라)**.

### 정규분포 가정 시 구체적 전개

$$
p(x\mid z_q(x)) = \mathcal{N}\big(x; \mu(z_q(x)), \sigma^2 I\big)
$$

$$
\log p(x\mid z_q(x)) = -\frac{1}{2\sigma^2}\lVert x - \mu(z_q(x)) \rVert^2 + \text{상수}
$$

$\sigma^2$ 고정, 상수 무시:

$$
\log p(x\mid z_q(x)) \propto -\lVert x - \mu(z_q(x)) \rVert^2
$$

→ **음의 MSE**. "$\log p(x\mid z_q(x))$를 최대화" = "MSE를 최소화".

### "재구성 loss"라고 부르는 이유

$\mu(z_q(x))$는 디코더의 출력, 즉 디코더가 $z_q(x)$로부터 복원해낸 결과 $\hat{x}$. $-\lVert x - \hat{x}\rVert^2$는 정확히 "원본과 복원본이 얼마나 다른가"를 재는 항 → 압축 전후가 얼마나 같은지 채점하는 항이므로 reconstruction loss라 부름.

### ELBO 최대화 vs Loss 최소화의 부호

ELBO는 최대화 대상, loss는 보통 최소화. 실제 구현:

$$
\text{ELBO 최대화} \Longleftrightarrow \text{$-$ELBO 최소화}
$$

실무에서는 정규분포 가정 + 고정분산이면 보통 직접 MSE로 구현:

```python
reconstruction_loss = MSE(x, decoder_output)
```

### 누구를 업데이트하는가 — 그래디언트 경로

$$
\text{Loss} = -\lVert x - \hat{x} \rVert^2,\qquad \hat{x} = \mathrm{Decoder}_{\theta_{\text{dec}}}(z_q(x)),\qquad z_q(x) = e_{k^\ast}
$$

```
Loss
 ↑
x̂ = Decoder(z_q(x))      ← 디코더 파라미터 θ_dec 가 여기
 ↑
z_q(x) = e_k              ← (forward에서만, 미분 안 됨)
 ↑
[straight-through: 그래디언트를 z_e(x)로 복사]
 ↑
z_e(x) = Encoder(x)       ← 인코더 파라미터 θ_enc 가 여기
```

| 대상 | 업데이트 여부 | 경로 |
|---|---|---|
| 디코더 $\theta_{\text{dec}}$ | ✓ 업데이트됨 | 직접 체인룰 |
| 인코더 $\theta_{\text{enc}}$ | ✓ 업데이트됨 | straight-through estimator 경유 |
| codebook $e$ | ✗ 업데이트 안 됨 | 그래디언트 경로 자체가 없음 (항 2에서만 업데이트) |

---

## 14. VQ Loss 항 상세 분석

### 식

$$
\lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2
$$

### sg 위치 확인

$$
\mathrm{sg}[z_e(x)] \to \text{동결} \qquad e \to \text{그래디언트 받음}
$$

$$
\frac{\partial}{\partial e}\lVert \mathrm{sg}[z_e(x)] - e \rVert^2 = -2\big(\mathrm{sg}[z_e(x)] - e\big)
$$

sg[$z_e(x)$]는 backward 시 상수 취급되므로 이 미분은 정상적으로 계산되고 $e$만 업데이트하는 그래디언트가 나옴.

### 직관: K-means clustering과 동일한 구조

이 항은 dictionary learning의 가장 단순한 형태인 Vector Quantisation(VQ)이라 불리며, 이는 K-means clustering과 본질적으로 동일.

K-means 절차:
1. 데이터 포인트를 가장 가까운 centroid에 할당
2. 각 centroid를 자신에게 할당된 데이터 포인트들의 **평균**으로 이동
3. 반복

VQ-VAE도 동일:
1. (식 2) $z_e(x)$를 가장 가까운 codebook 벡터 $e_k$에 할당 — "clustering 할당" 단계
2. (식 2의 VQ loss) $e_k$를 자신에게 할당된 $z_e(x)$들 쪽으로 이동 — "centroid 업데이트" 단계

### Appendix A.1: 닫힌 형태(closed-form) 최적해

codebook 항목 $e_i$에 가장 가깝게 할당된 인코더 출력들의 집합을 $\{z_{i,1}, \dots, z_{i,n_i}\}$라 하면, loss는:

$$
\sum_{j=1}^{n_i} \lVert z_{i,j} - e_i \rVert_2^2
$$

이 loss를 최소화하는 $e_i$의 닫힌 형태 최적해는 단순히 그 집합의 평균:

$$
e_i = \frac{1}{n_i}\sum_{j=1}^{n_i} z_{i,j}
$$

이것이 K-means에서 흔히 쓰이는 업데이트 방식.

### "L2 error"라고 부르는 이유

VQ objective는 $l_2$ error를 사용해 임베딩 벡터 $e_i$를 인코더 출력 $z_e(x)$ 쪽으로 이동시킨다. $\lVert \mathrm{sg}[z_e(x)] - e \rVert^2$는 두 점 사이 유클리드 거리의 제곱이고, 이것이 끌어당기는 힘의 세기를 정의 (거리가 멀수록 그래디언트가 커짐, 거리에 비례).

### EMA 대안 (Appendix A.1)

미니배치 학습에서는 한 번에 전체 평균을 계산할 수 없으므로, 이 loss term 대신 dictionary 항목을 exponential moving average(EMA)로 업데이트할 수도 있다 (이 논문 실험에서는 사용 안 함).

$$
N_i^{(t)} := N_i^{(t-1)}\cdot\gamma + n_i^{(t)}(1-\gamma)
$$
$$
m_i^{(t)} := m_i^{(t-1)}\cdot\gamma + \sum_j z_{i,j}^{(t)}(1-\gamma)
$$
$$
e_i^{(t)} := \frac{m_i^{(t)}}{N_i^{(t)}}
$$

$\gamma=0.99$가 실험적으로 잘 작동함을 확인 (이 논문 자체 실험에서는 안 썼지만, 후속 연구(VQGAN, TIGER/RQ-VAE 등)에서 널리 쓰임).

### 정리표

| | VQ loss (식 3의 항 2) |
|---|---|
| 업데이트 대상 | codebook $e$만 |
| 동결되는 것 | 인코더 (sg로 막힘) |
| 직관 | K-means의 centroid 업데이트와 동일 |
| 대안 구현 | gradient 대신 EMA로 직접 평균 갱신 |

---

## 15. Stop-gradient의 위치가 갖는 의미

### 항 2와 항 3 비교

$$
\text{항 2 (VQ loss)}: \quad \lVert \mathrm{sg}[z_e(x)] - e \rVert^2
$$
$$
\text{항 3 (commitment loss)}: \quad \lVert z_e(x) - \mathrm{sg}[e] \rVert^2
$$

겉보기에 거의 동일($z_e(x)$와 $e$ 사이 거리 제곱). 차이는 sg가 어느 쪽에 붙는가뿐:

$$
\text{항 2: sg가 } z_e(x)\text{에 붙음} \Rightarrow z_e(x)\text{ 동결, } e\text{는 자유} \Rightarrow e\text{만 업데이트}
$$
$$
\text{항 3: sg가 } e\text{에 붙음} \Rightarrow e\text{ 동결, } z_e(x)\text{는 자유} \Rightarrow z_e(x)\text{(인코더)만 업데이트}
$$

### 왜 두 항으로 나눠서 반대로 만들었나

$z_e(x)$와 $e$를 가깝게 만들고 싶다는 목표는 같지만, 방법은 두 가지:

1. $e$를 $z_e(x)$ 쪽으로 옮긴다 (codebook이 인코더를 따라가게)
2. $z_e(x)$를 $e$ 쪽으로 옮긴다 (인코더가 codebook을 따라가게)

만약 sg 없이 $\lVert z_e(x) - e\rVert^2$ 하나로 뒀다면 양쪽이 동시에 움직여서 서로 쫓아가는 모양이 됨 — moving target problem으로 학습이 불안정해질 수 있음. 그래서 같은 거리 항을 두 번 따로 만들어서 각각 한쪽만 움직이게 분리:

```
항 2: "e만 움직여라 (z_e(x)는 가만히 있다고 치고)"  → e를 위한 학습 신호
항 3: "z_e(x)만 움직여라 (e는 가만히 있다고 치고)"  → 인코더를 위한 학습 신호
```

### 비유

밧줄 양 끝에 사람 A(인코더)와 사람 B(codebook)가 가까워지길 원할 때:
- 항 2: "B만 A 쪽으로 걸어가라, A는 그 자리에 서 있어라"
- 항 3: "A만 B 쪽으로 걸어가라, B는 그 자리에 서 있어라"

두 스텝을 동시에(같은 backward pass에서) 계산하지만 서로의 그래디언트에 영향을 주지 않게 분리 → 한 번의 업데이트로 양쪽이 동시에, 서로 헷갈리지 않게 가까워짐.

### sg가 없었다면?

$\lVert z_e(x) - e\rVert^2$ (sg 없이)를 $z_e(x)$로 미분하면 $+2(z_e(x)-e)$, $e$로 미분하면 $-2(z_e(x)-e)$ — 둘이 정확히 반대 방향으로 동시에 움직임. 이것도 작동할 수 있지만, 두 항으로 분리하면 각각 다른 가중치(항 2는 가중치 1, 항 3은 $\beta=0.25$)를 줄 수 있어 "codebook은 빠르게, 인코더는 천천히/약하게" 같은 **비대칭적 조절**이 가능해짐 — 이것이 두 항으로 나눈 진짜 이유.

### 정리표

| | 누가 움직이나 | sg가 막는 대상 | 비유 |
|---|---|---|---|
| 항 2 (VQ) | codebook $e$ | 인코더 $z_e(x)$ | "codebook아, 인코더 따라가" |
| 항 3 (commitment) | 인코더 $z_e(x)$ | codebook $e$ | "인코더야, codebook에 전념해" |

---

## 16. VQ Loss / Commitment Loss가 왜 필요한가 (근본 이유)

### 만약 항 2, 3이 없다면?

reconstruction loss만 있다고 가정:

$$
\mathcal{L} = \log p(x\mid z_q(x)) \quad \text{(이것만)}
$$

그래디언트 경로:

$$
\text{Loss} \to \hat{x} = \mathrm{Decoder}(z_q(x)) \to z_q(x)=e_k \to [\text{straight-through: 점프}] \to z_e(x)=\mathrm{Encoder}(x)
$$

$e_k$ 자체로 거슬러 올라가는 그래디언트 경로가 없음 — straight-through estimator는 "$z_q(x)$에서 받은 그래디언트를 $z_e(x)$로 그냥 복사"하는 역할만 하지, $e_k$를 업데이트하는 경로는 만들지 않음.

**결론: reconstruction loss만 있으면 codebook $e$는 한 번도 업데이트되지 않고 초기화된 값에 멈춤.**

### codebook은 반드시 학습되어야 한다

codebook $e$는 모델의 핵심 파라미터 — 모든 입력 $x$가 $K$개 벡터 중 하나로 매핑되어야 하는데, 이 벡터들이 랜덤 초기값에 멈춰있으면 의미 있는 quantization이 절대 안 됨.

AE와 VQ-VAE의 근본적 차이: AE는 모든 파라미터가 하나의 매끄러운 미분 가능 경로로 연결되어 reconstruction loss 하나로 전부 학습됨. VQ-VAE는 argmin이라는 미분 불가능 연산이 끼어있어 그 양쪽(인코더 쪽, codebook 쪽)에 각각 별도의 학습 신호를 만들어줘야 함.

### 항 2가 등장하는 이유

straight-through gradient estimation으로 인해 임베딩 $e_i$는 reconstruction loss로부터 그래디언트를 전혀 받지 못하므로, 임베딩 공간을 학습시키기 위해 가장 단순한 dictionary learning 알고리즘인 Vector Quantisation(VQ)을 사용한다. → reconstruction loss의 구조적 빈틈(codebook 미업데이트)을 메우기 위해 의도적으로 추가된 항.

### 항 3(commitment loss)이 또 필요한 이유

embedding space의 크기는 차원이 없어서(dimensionless), embedding이 인코더 파라미터만큼 빠르게 학습되지 않으면 임의로 커질 수 있다.

구체적으로: 항 1(reconstruction)은 인코더가 "디코더가 복원하기 좋은 $z_e(x)$"를 내놓도록 압박. 만약 인코더가 codebook과 점점 멀어지는 방향으로 $z_e(x)$를 키워도 reconstruction이 잘 된다면, 항 2가 codebook을 그 멀어진 자리까지 쫓아가게 만들 수 있음 → 둘이 함께 발산(diverge)할 위험. 이를 막기 위해 "인코더야, 가까운 codebook 벡터 근처에 확실히 정착(commit)해라"는 별도 압박이 필요 — 이것이 항 3.

### 흐름 요약

```
목표: discrete codebook을 가진 모델을 학습시키고 싶다
        ↓
문제 1: argmin이 미분 안 됨 → straight-through로 인코더 쪽은 우회
        ↓
문제 2: 그 우회 때문에 codebook(e)이 전혀 그래디언트를 못 받음
        ↓ 해결: 항 2 추가 (codebook을 위한 전용 학습 신호)
        ↓
문제 3: 인코더와 codebook이 서로 발산할 위험
        ↓ 해결: 항 3 추가 (인코더를 codebook 근처에 묶어두는 신호)
```

### 왜 하필 $\lVert \mathrm{sg}[z_e(x)] - e \rVert^2$ 형태인가 (e만 단독으로는 안 되는 이유)

**$e$만 단독으로 들어간 항이라면?** 예를 들어 $\lVert e \rVert^2$ — 이건 $e$의 크기를 작게 만드는 정규화 항일 뿐, "어디로" 가야 하는지에 대한 정보가 전혀 없음.

**핵심: $e$를 학습시키려면 목표 지점(target)이 있어야 한다.** $e_k$의 목표는 "이 $e_k$에 할당된 인코더 출력들과 비슷해져야 한다"는 것. codebook이 "데이터가 실제로 어떤 모양으로 분포해 있는지"를 대표하길 원하므로, $e_k$가 따라가야 할 목표는 정확히 그 $e_k$로 매핑되는 $z_e(x)$들. 그래서 항은 반드시 "$e$와 $z_e(x)$ 사이의 거리" 형태여야 함 — $z_e(x)$가 목표(target), $e$가 그 목표를 따라가는 변수.

**왜 거리는 L2인가**: L2 loss의 최적해는 정확히 평균(mean) (Appendix A.1). L1 거리(절댓값)를 썼다면 최적해는 중앙값(median). K-means와 같은 평균 기반 clustering을 원했기 때문에 L2를 택함.

**왜 sg가 $z_e(x)$ 쪽인가**: sg 없이 $\lVert z_e(x)-e\rVert^2$라면 인코더와 codebook이 동시에 서로 쫓아가는 moving target 문제 발생. K-means의 "할당 → centroid 업데이트"를 번갈아가며 하는 절차를 모사하려면 "이 스텝에서는 $z_e(x)$를 고정된 데이터처럼 취급하고 $e$만 그쪽으로 이동"시켜야 함 — sg[$z_e(x)$]가 정확히 이 역할.

### 정리표

| 질문 | 답 |
|---|---|
| 왜 $e$만 단독으로 안 쓰나 | $e$가 향해야 할 "목표"가 없으면 학습 신호가 안 나옴. 목표는 $z_e(x)$여야 함 |
| 왜 거리는 L2인가 | 최적해가 평균(mean)이 되도록, K-means식 centroid 업데이트를 모사 |
| 왜 sg가 $z_e(x)$에 붙나 | "$z_e(x)$는 고정된 타겟, $e$만 그쪽으로 이동"이라는 K-means식 분리된 업데이트를 위함 |

---

## 17. 구체적 분포 가정 하에서 ELBO → 식 3 전체 유도

### 설정

- **이미지 $x$**: $H\times W\times 3$ 픽셀의 실수 벡터
- **Codebook**: $K$개의 벡터 $e_1,\dots,e_K \in \mathbb{R}^D$
- **인코더**: $x \to z_e(x) \in \mathbb{R}^D$ deterministic (CNN)
- **Prior** $p(z)$: $K$개 카테고리에 대한 균등분포

$$
p(z=k) = \frac{1}{K} \quad \text{for all } k
$$

- **Posterior** $q(z\mid x)$ (식 1): one-hot

$$
q(z=k\mid x) = \mathbb{1}[k=k^\ast], \qquad k^\ast = \arg\min_j \lVert z_e(x)-e_j \rVert
$$

- **Decoder** $p(x\mid z)$: 등방성(isotropic) 정규분포

$$
p(x\mid z=k) = \mathcal{N}\big(x; D(e_k), \sigma^2 I\big)
$$

$D(\cdot)$는 디코더 신경망, 분산 $\sigma^2$는 고정 상수.

### 1단계: ELBO를 이 설정에 대입

일반 ELBO:

$$
\mathrm{ELBO} = \mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big] - \mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z)\big)
$$

**기댓값 항 풀기**: $q(z\mid x)$가 one-hot이므로 $K$개 항의 합이 단 하나의 항으로 붕괴:

$$
\mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big] = \sum_k q(z=k\mid x)\log p(x\mid z=k) = 1\cdot\log p(x\mid z=k^\ast) = \log p(x\mid z_q(x))
$$

여기서 $z_q(x) = e_{k^\ast}$. → **식 3의 항 1이 정확히 여기서 나옴** — q(z|x)가 one-hot이라는 가정 때문에 기댓값이 선택된 하나의 항으로 붕괴.

**KL 항**: [9번 섹션](#9-kl-divergence가-상수-log-k가-되는-이유-직접-계산)에서 유도한 대로:

$$
\mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z)\big) = \log K \quad \text{(상수, 그래디언트 0이므로 제거)}
$$

**따라서**:

$$
\mathrm{ELBO} = \log p(x\mid z_q(x)) - \log K \approx \log p(x\mid z_q(x)) \quad (\log K \text{ 무시 가능})
$$

이것이 식 3의 항 1.

### 2단계: $\log p(x\mid z_q(x))$를 정규분포 가정으로 구체화

$$
p(x\mid z_q(x)) = \mathcal{N}\big(x; D(z_q(x)), \sigma^2 I\big)
$$

다변량 정규분포의 log-density (차원 수 $n$):

$$
\log p(x\mid z_q(x)) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\lVert x - D(z_q(x)) \rVert^2
$$

$\sigma^2$를 고정 상수로 두면 $x$에 대해 변하는 부분은 두 번째 항뿐:

$$
\log p(x\mid z_q(x)) = -\frac{1}{2\sigma^2}\lVert x - \hat{x} \rVert^2 + \text{상수}, \qquad \hat{x} = D(z_q(x))
$$

→ reconstruction loss가 MSE인 이유의 정확한 수식적 근거: $-\mathrm{ELBO}$ 최소화 = $\frac{1}{2\sigma^2}\lVert x-\hat{x}\rVert^2$ 최소화 = ($\sigma^2$ 무시하면) MSE 최소화.

### 3단계: 전체 negative ELBO (실제 학습 loss)

$$
-\mathrm{ELBO} = \frac{1}{2\sigma^2}\lVert x - D(z_q(x)) \rVert^2 + \log K + [\text{학습에 안 쓰임, 상수}]
$$

여기까지는 순수 ELBO에서 유도된 부분. 하지만 이 항만으로는 codebook $e$가 학습이 안 됨([16번 섹션](#16-vq-loss--commitment-loss가-왜-필요한가-근본-이유) 참고). 그래서 **ELBO 바깥에서** VQ loss와 commitment loss를 보조 항으로 추가:

$$
\mathcal{L} = \underbrace{\frac{1}{2\sigma^2}\lVert x - D(z_q(x)) \rVert^2}_{\text{ELBO에서 유도됨 (reconstruction)}} + \underbrace{\lVert \mathrm{sg}[z_e(x)] - e_{k^\ast} \rVert^2}_{\text{ELBO 밖에서 추가 (codebook 학습용)}} + \underbrace{\beta\lVert z_e(x) - \mathrm{sg}[e_{k^\ast}] \rVert^2}_{\text{ELBO 밖에서 추가 (commitment)}}
$$

논문 식 3에서 $\frac{1}{2\sigma^2}$을 그냥 1로 흡수해 표기한 것뿐이고, 본질적으로 같은 형태.

### 4단계: $N$개 latent에 대한 확장

실제로는 $z$가 하나가 아니라 $32\times32$ 같은 grid 전체 (각 위치마다 독립적인 discrete latent). $N$개의 latent가 있다면:

$$
\mathcal{L} = \frac{1}{2\sigma^2}\Big\lVert x - D\big(\{z_q(x)\}_{1..N}\big) \Big\rVert^2 + \frac{1}{N}\sum_{i=1}^{N}\lVert \mathrm{sg}[z_{e,i}(x)] - e_{k_i} \rVert^2 + \frac{\beta}{N}\sum_{i=1}^{N}\lVert z_{e,i}(x) - \mathrm{sg}[e_{k_i}] \rVert^2
$$

논문: 결과 loss $L$은 동일하지만, k-means와 commitment loss에 대해서는 $N$개 항의 평균을 취하게 된다 — reconstruction은 전체 이미지에 대해 한 번, 나머지 둘은 $N$개 latent 위치마다 평균.

### 전체 유도 흐름 요약

```
일반 VAE의 ELBO
   │ (q(z|x) = one-hot 대입)
   ▼
ELBO = log p(x|z_q(x)) - log K
   │ (log K는 상수라 무시, p(x|z)를 정규분포로 구체화)
   ▼
-ELBO ∝ ||x - x̂||²   (MSE = reconstruction loss, 식 3의 항 1)
   │ (그런데 이것만으론 e가 학습 안 됨 — straight-through의 빈틈)
   ▼
+ VQ loss (codebook용, ELBO 밖에서 추가)
+ commitment loss (인코더 안정화용, ELBO 밖에서 추가)
   │
   ▼
식 3 완성
```

→ 식 3의 첫 항은 정말로 ELBO에서 곧바로 떨어지는 항이고, 나머지 두 항은 discrete quantization이라는 구조가 만든 미분 불가능성을 보완하기 위해 ELBO 바깥에서 추가된 engineering적 장치.

---

## 18. 각 파라미터에 대한 Gradient 유도

### 표기 정리

$$
z_e(x) = \mathrm{Enc}_{\theta_{\text{enc}}}(x) \in \mathbb{R}^D
$$
$$
k^\ast = \arg\min_j \lVert z_e(x) - e_j \rVert \quad \text{(forward에서 결정되는 정수, 미분 대상 아님)}
$$
$$
z_q(x) = e_{k^\ast}
$$
$$
\hat{x} = \mathrm{Dec}_{\theta_{\text{dec}}}(z_q(x))
$$

세 항 (단순화를 위해 $1/(2\sigma^2)$ 생략):

$$
\mathcal{L}_1 = \lVert x - \hat{x} \rVert^2 \qquad \mathcal{L}_2 = \lVert \mathrm{sg}[z_e(x)] - e_{k^\ast} \rVert^2 \qquad \mathcal{L}_3 = \beta\lVert z_e(x) - \mathrm{sg}[e_{k^\ast}] \rVert^2
$$

### 1) 디코더 파라미터 $\theta_{\text{dec}}$

$\mathcal{L}_2, \mathcal{L}_3$에는 디코더가 등장하지 않으므로 그래디언트는 0. $\mathcal{L}_1$만:

$$
\frac{\partial \mathcal{L}_1}{\partial \theta_{\text{dec}}} = \frac{\partial \mathcal{L}_1}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \theta_{\text{dec}}} = -2(x-\hat{x}) \cdot \frac{\partial \mathrm{Dec}_{\theta_{\text{dec}}}(z_q(x))}{\partial \theta_{\text{dec}}}
$$

평범한 신경망 backprop — $z_q(x)$를 고정 입력값으로 보고 디코더 내부를 표준적으로 미분.

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \theta_{\text{dec}}} = -2(x-\hat{x})\cdot\frac{\partial \mathrm{Dec}}{\partial \theta_{\text{dec}}}}
$$

### 2) Codebook $e_{k^\ast}$에 대한 그래디언트

**$\mathcal{L}_1$에서**: straight-through estimator 정의상 $e_{k^\ast}$ 자체로는 그래디언트가 전달되지 않음.

$$
\frac{\partial \mathcal{L}_1}{\partial e_{k^\ast}} = 0
$$

**$\mathcal{L}_2$에서**: $\mathrm{sg}[z_e(x)]$는 상수 취급, $e_{k^\ast}$만 미분 대상.

$$
\mathcal{L}_2 = \lVert \mathrm{sg}[z_e(x)] - e_{k^\ast}\rVert^2 = \sum_d (z_e(x)_d - e_{k^\ast,d})^2
$$
$$
\frac{\partial \mathcal{L}_2}{\partial e_{k^\ast}} = 2(e_{k^\ast} - z_e(x))
$$

**$\mathcal{L}_3$에서**: $\mathrm{sg}[e_{k^\ast}]$가 상수이므로 $e_{k^\ast}$로 미분하면 0.

$$
\frac{\partial \mathcal{L}_3}{\partial e_{k^\ast}} = 0
$$

**다른 모든 $e_j\ (j\ne k^\ast)$**: 어느 항에도 등장하지 않으므로 그래디언트 0.

**합산**:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial e_{k^\ast}} = 2(e_{k^\ast} - z_e(x)), \qquad \frac{\partial \mathcal{L}}{\partial e_j} = 0 (j\ne k^\ast)}
$$

그래디언트 디센트 업데이트:

$$
e_{k^\ast} \leftarrow e_{k^\ast} - \eta\cdot 2(e_{k^\ast}-z_e(x)) = e_{k^\ast} - 2\eta(e_{k^\ast}-z_e(x))
$$

$e_{k^\ast}-z_e(x)$만큼 빼주므로 $e_{k^\ast}$가 $z_e(x)$ 쪽으로 이동 — "e가 z_e(x) 쪽으로 끌려간다"는 직관이 수식으로 확인됨.

### 3) 인코더 파라미터 $\theta_{\text{enc}}$

$z_e(x) = \mathrm{Enc}_{\theta_{\text{enc}}}(x)$이므로 체인룰의 마지막 단계는 항상 $\partial z_e(x)/\partial\theta_{\text{enc}}$.

**$\mathcal{L}_1$에서 (straight-through 적용)**

정상적인 체인룰:

$$
\frac{\partial \mathcal{L}_1}{\partial z_e(x)} = \frac{\partial \mathcal{L}_1}{\partial \hat{x}}\cdot\frac{\partial \hat{x}}{\partial z_q(x)}\cdot\frac{\partial z_q(x)}{\partial z_e(x)}
$$

마지막 항 $\partial z_q(x)/\partial z_e(x)$는 argmin 때문에 미정의. **Straight-through estimator는 이를 항등(identity)으로 강제 정의**:

$$
\frac{\partial z_q(x)}{\partial z_e(x)} := I
$$

따라서:

$$
\frac{\partial \mathcal{L}_1}{\partial z_e(x)} := \frac{\partial \mathcal{L}_1}{\partial \hat{x}}\cdot\frac{\partial \hat{x}}{\partial z_q(x)} = -2(x-\hat{x})\cdot\frac{\partial \mathrm{Dec}(z_q(x))}{\partial z_q(x)}
$$

(디코더를 $z_q(x)$로 미분한 야코비안을, 마치 $z_e(x)$에서 계산한 것처럼 재사용)

**$\mathcal{L}_2$에서**: $\mathrm{sg}[z_e(x)]$가 상수이므로 $z_e(x)$로 미분하면 0.

$$
\frac{\partial \mathcal{L}_2}{\partial z_e(x)} = 0
$$

**$\mathcal{L}_3$에서**: $z_e(x)$만 미분 대상, $\mathrm{sg}[e_{k^\ast}]$는 상수.

$$
\mathcal{L}_3 = \beta\lVert z_e(x) - \mathrm{sg}[e_{k^\ast}] \rVert^2 = \beta\sum_d (z_e(x)_d - e_{k^\ast,d})^2
$$
$$
\frac{\partial \mathcal{L}_3}{\partial z_e(x)} = 2\beta(z_e(x) - e_{k^\ast})
$$

**합산 (체인룰 전 단계)**:

$$
\frac{\partial \mathcal{L}}{\partial z_e(x)} = \underbrace{-2(x-\hat{x})\cdot\frac{\partial \mathrm{Dec}}{\partial z_q(x)}}_{\mathcal{L}_1 \text{ (ST 경유)}} + \underbrace{2\beta(z_e(x)-e_{k^\ast})}_{\mathcal{L}_3 \text{ (commitment)}}
$$

**최종, $\theta_{\text{enc}}$까지 체인룰 완성**:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \theta_{\text{enc}}} = \left[-2(x-\hat{x})\cdot\frac{\partial \mathrm{Dec}}{\partial z_q(x)} + 2\beta(z_e(x)-e_{k^\ast})\right]\cdot\frac{\partial z_e(x)}{\partial \theta_{\text{enc}}}}
$$

인코더는 첫 번째와 마지막 loss term을 최적화한다는 논문 문장이 수식으로 확인됨 — 인코더가 받는 그래디언트는 reconstruction 경유분 + commitment 경유분 두 조각으로 구성.

### 최종 정리표

| 파라미터 | $\mathcal{L}_1$ (reconstruction) | $\mathcal{L}_2$ (VQ) | $\mathcal{L}_3$ (commitment) | 최종 그래디언트 |
|---|---|---|---|---|
| $\theta_{\text{dec}}$ | 정상 체인룰 | 0 | 0 | $\mathcal{L}_1$만 |
| $e_{k^\ast}$ | 0 (ST가 끊음) | $2(e_{k^\ast}-z_e(x))$ | 0 (sg) | $\mathcal{L}_2$만 |
| $\theta_{\text{enc}}$ | ST로 우회 전달 | 0 (sg) | $2\beta(z_e(x)-e_{k^\ast})$ | $\mathcal{L}_1$(ST) + $\mathcal{L}_3$ |

세 파라미터 그룹이 서로 겹치지 않는 깔끔한 분업 구조를 이룸 — sg를 정확히 그 자리에 배치한 설계 의도가 그래디언트 경로를 깔끔하게 분리시키는 것.

---

## 19. $N$개 latent로의 확장과 $\log p(x)$ 평가

### $N$개 discrete latent

실험에서는 $N$개의 discrete latent를 정의 (예: ImageNet에는 $32\times32$ latent field, CIFAR10에는 $8\times8\times10$). 전체 loss는 동일하지만, k-means(VQ) loss와 commitment loss는 $N$개 항에 대한 평균이 됨. (수식은 [17번 섹션 4단계](#17-구체적-분포-가정-하에서-elbo--식-3-전체-유도) 참고)

### $\log p(x)$의 정확한 marginal likelihood

$$
\log p(x) = \log\sum_k p(x\mid z_k)p(z_k)
$$

$z$가 discrete categorical이므로 적분이 아니라 $K$개 항에 대한 합으로 정확히 표현됨. 의미: "$x$가 나올 확률은, $z$가 각 카테고리 $k$일 때 디코더가 $x$를 만들어낼 확률 $p(x\mid z_k)$에 그 카테고리가 선택될 확률 $p(z_k)$를 곱해서 모든 $k$에 대해 합한 것."

### 문제: 계산 불가능

$K=512$, $N=1024$ 같은 grid 전체에 대해서는 $K^N$개 조합을 다 더해야 함 — 천문학적으로 계산 불가능.

### 해법: "디코더가 학습 후엔 $z_q(x)$ 외 다른 $z$에는 확률을 안 준다"는 근사

디코더 $p(x\mid z)$는 MAP-inference로 얻은 $z=z_q(x)$로 학습되었기 때문에, 완전히 수렴한 후에는 $z\ne z_q(x)$인 경우 $p(x\mid z)$에 확률 질량을 전혀 할당하지 않을 것이다.

- 학습 과정 전체에서 디코더는 항상 $z_q(x)$(그 $x$에 대해 실제로 선택된 codebook 인덱스)만 입력으로 받아 그 $x$를 복원하도록 훈련됨
- $z_q(x)$가 아닌 다른 인덱스 $k'$를 디코더에 넣으면, 디코더는 그 입력에 대해 "이 $x$를 만들어내라"고 학습받은 적이 없음
- 학습이 충분히 수렴했다면 $p(x\mid z=k')$는 $k'\ne k^\ast$일 때 거의 0

즉 $K$개 항의 합에서 사실상 $z_q(x)$에 해당하는 항 하나만 의미 있는 크기를 갖고 나머지는 거의 0.

### 결과적으로 합이 한 항으로 붕괴

$$
\log p(x) = \log\sum_k p(x\mid z_k)p(z_k) \approx \log\big[p(x\mid z_q(x))\cdot p(z_q(x))\big] = \log p(x\mid z_q(x)) + \log p(z_q(x))
$$

이것이 논문의 근사식 $\log p(x) \approx \log p(x\mid z_q(x))p(z_q(x))$. Jensen's inequality로 이것이 사실은 하한($\ge$)임도 명시됨:

$$
\log p(x) \ge \log p(x\mid z_q(x))p(z_q(x))
$$

### 학습 시 ELBO 유도와의 차이점

| | 학습 시 ELBO 유도 | 평가 시 $\log p(x)$ 근사 |
|---|---|---|
| 근거 | $q(z\mid x)$가 one-hot이라서 기댓값 합이 *정의상* 한 항으로 줄어듦 | 디코더가 다른 $z$에 확률을 거의 0으로 학습했을 거라는 *경험적 근사* |
| 정확성 | 수학적으로 정확 | 근사 (학습이 잘 됐다면 타당) |

후자는 근사이기 때문에 논문에서 "empirically evaluate this approximation"(4절에서 실험적으로 검증)이라는 표현이 따라붙음.

---

## 20. 3.3 Prior

### 핵심 문장

prior $p(z)$는 categorical 분포이고, feature map 내 다른 $z$들에 의존하게 만들면 autoregressive하게 만들 수 있다. VQ-VAE를 학습하는 동안 prior는 고정된 균등분포로 유지되고, 학습이 끝난 후에 비로소 discrete latent $z$에 대한 autoregressive 분포 $p(z)$를 따로 fit해서, ancestral sampling으로 $x$를 생성할 수 있게 만든다.

### 왜 이게 "2단계"인가

학습 중에는 ($\S$9, $\S$17에서 확인):

$$
p(z) = \text{uniform (고정)}, \qquad \mathrm{KL}\big(q(z\mid x)\mathrel{\Vert} p(z)\big) = \log K \text{ (상수, 그래디언트 0)}
$$

즉 학습 단계에서 prior는 전혀 학습되지 않음. 학습이 끝나면 인코더, 디코더, codebook이 모두 고정되고, 그 다음에야 진짜 prior를 학습하는 별도의 두 번째 학습 단계가 시작됨.

```
[1단계: VQ-VAE 학습]                    [2단계: Prior 학습]
인코더 + 디코더 + codebook 학습         인코더/디코더/codebook 전부 고정
p(z) = uniform (고정, 학습 안 함)        p(z)를 PixelCNN/WaveNet으로 학습
목표: 좋은 discrete 표현 만들기          목표: 그 표현 위에서 생성 가능하게 만들기
```

일반 VAE는 $p(z)=\mathcal{N}(0,I)$로 고정한 채 끝 (학습 후 바로 prior에서 샘플링 가능). VQ-VAE는 1단계가 끝난 후 prior 자체를 새로 학습해야 생성이 가능해짐.

### 왜 prior를 따로 학습해야 하는가

학습 중 $p(z)=$ uniform이라는 가정은 "어떤 codebook 인덱스든 똑같이 나올 법하다"는 뜻인데, 실제로는 전혀 그렇지 않음 (예: $32\times32$ latent grid에서 인접한 위치의 latent끼리는 강한 상관관계). uniform prior는 이런 구조를 전혀 반영하지 못함.

학습이 끝난 후, 실제로 학습 데이터들이 codebook 인덱스 grid 상에서 어떤 패턴으로 나타나는지를 또 다른 모델로 학습 — 이것이 **autoregressive하게 만든다**는 부분. $z$의 그리드를 순서대로 한 칸씩 보면서, 이전 칸들을 보고 다음 칸의 인덱스를 예측하는 모델:

$$
p(z) = p(z_1)\cdot p(z_2\mid z_1)\cdot p(z_3\mid z_1,z_2)\cdots p(z_N\mid z_1,\dots,z_{N-1})
$$

이미지에는 **PixelCNN**, 음성에는 **WaveNet**을 이 두 번째 학습 단계에 사용.

### Ancestral sampling으로 생성하기

1. $p(z_1)$에서 $z_1$ 샘플링
2. $p(z_2\mid z_1)$에서 $z_2$ 샘플링
3. ... 차례로 $z_N$까지
4. 완성된 latent grid $(z_1,\dots,z_N)$을 디코더에 넣어 이미지로 변환

각 변수를 그 "조상(이전 변수들)"이 정해진 후에 차례로 뽑는 방식이라 ancestral sampling.

### Future work로 남긴 부분

prior와 VQ-VAE를 동시에(jointly) 학습하는 것이 결과를 더 강화할 수 있겠지만, 이는 future research로 남긴다.

저자들도 2단계로 분리한 게 최선은 아닐 수 있다는 걸 인지. 이 지점이 후속 연구(VQGAN, TIGER 등 semantic ID 계열)에서 계속 개선되는 부분 — 예: TIGER 계열은 VQ-VAE/RQ-VAE로 만든 discrete code(semantic ID) 위에 별도의 시퀀스 모델(Transformer)을 얹는 식으로, 본질적으로 같은 2단계 구조를 유지하면서 prior 역할을 하는 모델을 더 강력하게(Transformer 기반) 만든 것.

### 정리표

| | 1단계 (VQ-VAE 학습) | 2단계 (Prior 학습) |
|---|---|---|
| 학습 대상 | 인코더, 디코더, codebook | autoregressive 모델 (PixelCNN/WaveNet) |
| $p(z)$ | uniform, 고정 | 데이터 기반으로 학습됨 |
| 목적 | 좋은 discrete 표현 학습 | 그 표현 위에서 생성 가능하게 |
| 이미지 적용 모델 | CNN 인코더/디코더 | PixelCNN |
| 음성 적용 모델 | WaveNet 디코더 | WaveNet |

---

## 부록: 다음 공부 방향 (미진행)

- 4절 실험 (4.1 continuous와의 비교, 4.2 이미지, 4.3 오디오, 4.4 비디오)
- RQ-VAE / TIGER (Semantic ID, arXiv:2403.08206) — VQ-VAE의 discrete code를 추천시스템 아이템 ID로 확장
- VQGAN, EMA 기반 codebook 업데이트 실전 구현
