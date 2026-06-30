# AE → VAE → VQ-VAE 계열 정리

AE, VAE, VQ-VAE, VQ-VAE-2, VQ-GAN, RQ-VAE를 거쳐 TIGER(Generative Retrieval)까지 이어지는 흐름을 정리한 노트.

## 1. 아키텍처 (간단 수식)

| 모델 | 흐름 |
|---|---|
| AE | $x \to z = f(x) \to \hat{x} = g(z)$ |
| VAE | $x \to q(z\|x) \to z \sim q(z\|x) \to p(x\|z)$ |
| VQ 계열 (VQ-VAE / VQ-VAE-2 / VQ-GAN / RQ-VAE) | $x \to z_e = f(x) \to z_q = \text{Quantize}(z_e) \to \hat{x} = g(z_q)$ |

- AE: 인코더/디코더 모두 결정적(deterministic) 함수
- VAE: 인코더가 분포 $q(z\vert x)$를 출력 → 샘플링 → 디코더가 $p(x\vert z)$ 복원
- VQ 계열: 인코더 출력 $z_e$가 codebook으로 양자화되는 단계 $\text{Quantize}(\cdot)$가 추가됨. 이 양자화 방식(단일/계층/잔차)이 모델별로 다름 (3번 축 참고)

## 2. Latent space 가정 / Loss 구성

| 모델 | Latent 종류 | Loss 구성 | 비고 |
|---|---|---|---|
| AE | 연속, 결정적 (deterministic) | Reconstruction만 | Prior 가정 없음, 생성 모델 아님 |
| VAE | 연속, 확률적 | Reconstruction + KL divergence | ELBO 최대화, posterior collapse 가능 |
| VQ-VAE | 이산, codebook 1개 | Reconstruction + commitment loss + codebook loss | Straight-through estimator로 미분 우회 |
| VQ-VAE-2 | 이산, 계층(공간 multi-scale) | VQ-VAE와 동일 구조 반복 | 고해상도 디테일 보강 |
| VQ-GAN | 이산, codebook 1개 | Reconstruction + commitment + codebook + perceptual loss + adversarial loss | Pixel loss의 blurry 문제를 GAN으로 보완 |
| RQ-VAE | 이산, 계층(깊이 residual) | VQ-VAE와 동일 구조 반복 | 적은 codebook 크기로 정밀도 확보 |

## 3. Quantization 방식

VQ 계열 내부를 가르는 핵심 축.

- **VQ-VAE**: 단일 codebook에서 nearest-neighbor lookup으로 1회 양자화
- **VQ-VAE-2**: 공간(spatial) 계층 구조 — top/bottom 두 해상도 레벨에서 각각 독립적인 codebook으로 양자화 (multi-scale)
- **VQ-GAN**: VQ-VAE와 동일한 단일 codebook NN lookup (quantization 자체는 변화 없음, loss/생성 모델만 다름)
- **RQ-VAE**: 같은 위치의 벡터를 residual 방식으로 반복 양자화 (quantize → residual → 재양자화 → ...). 공간이 아니라 **깊이(depth) 방향**으로 정밀도를 높임

> Codebook 크기/표현력 trade-off: VQ-VAE-2는 "공간을 더 세밀하게", RQ-VAE는 "깊이를 더 정밀하게" 풀어 codebook collapse 문제에 대응한다는 점이 대비됨.

## 4. 생성 시 사용하는 2단계 모델

AE/VAE는 1단계 모델에서 바로 생성(혹은 생성 불가)하지만, VQ 계열부터는 **1단계: 이산 latent로 압축 → 2단계: 이산 시퀀스 분포를 별도 모델로 학습**하는 two-stage 구조가 핵심.

| 모델 | 2단계 생성 모델 | 시퀀스 처리 방식 |
|---|---|---|
| VQ-VAE | PixelCNN | 2D spatial map을 raster-scan 순서로 autoregressive 처리 |
| VQ-VAE-2 | 계층별 PixelCNN (top prior + bottom conditional prior) | Top 먼저 생성 → top을 condition으로 bottom 생성 |
| VQ-GAN | Transformer (GPT-style, decoder-only) | Codebook index를 1D sequence로 flatten 후 autoregressive 생성 |
| RQ-VAE | RQ-Transformer (spatial transformer + depth transformer 2단 구조) | 공간 위치별로 residual depth 코드들을 추가로 autoregressive 생성 |

**핵심 흐름**: CNN 기반 raster-scan(PixelCNN) → 1D self-attention(Transformer, VQ-GAN) → spatial-depth factorized self-attention(RQ-Transformer)

PixelCNN은 spatial locality에 의존하는 inductive bias라 멀리 떨어진 위치 간 관계를 잘 못 잡는 반면, VQ-GAN의 Transformer 도입으로 global dependency를 self-attention으로 처리할 수 있게 됨. RQ-VAE는 한 위치당 여러 depth의 code가 생기는 문제를 spatial/depth transformer 분리로 해결.

## 5. TIGER의 위치

TIGER는 위 two-stage 패러다임을 **이미지 생성 → 추천(Generative Retrieval)** 도메인으로 이식한 연구.

| 단계 | 이미지 생성 계열 | TIGER |
|---|---|---|
| 압축 대상 | 이미지 픽셀 | 아이템 메타데이터(텍스트 임베딩) |
| 1단계 양자화 | RQ-VAE | RQ-VAE (그대로 재사용) |
| 2단계 모델 | RQ-Transformer (spatial+depth autoregressive) | Transformer encoder-decoder (seq2seq) |
| 2단계 목적 | 새 이미지 생성 (unconditional) | 유저 히스토리에 condition된 다음 추천 아이템(semantic ID) 생성 |

- 1단계: RQ-VAE로 아이템을 압축적인 이산 코드 시퀀스(semantic ID)로 표현
- 2단계: 유저의 과거 아이템 semantic ID 시퀀스를 입력으로, 다음 추천 아이템의 semantic ID 시퀀스를 생성 (seq2seq, RQ-Transformer의 unconditional 생성과 달리 user-conditioned)

기존 추천 시스템이 아이템을 랜덤 ID(embedding lookup)로 다뤘다면, TIGER는 이를 의미를 담은 semantic ID로 바꾸고 추천 자체를 "다음 아이템 ID 시퀀스를 생성하는" 문제로 재정의함.

## 참고 논문

- AE / VAE: Auto-Encoding Variational Bayes (Kingma & Welling, 2013)
- VQ-VAE: Neural Discrete Representation Learning (van den Oord et al., 2017)
- VQ-VAE-2: Generating Diverse High-Fidelity Images with VQ-VAE-2 (Razavi et al., 2019)
- VQ-GAN: Taming Transformers for High-Resolution Image Synthesis (Esser et al., 2021)
- RQ-VAE: Autoregressive Image Generation using Residual Quantization (Lee et al., 2022)
- TIGER: Recommender Systems with Generative Retrieval (Rajput et al., 2023)