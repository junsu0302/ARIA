# ARIA

Pure NumPy 기반 밑바닥부터 구현한 고속 딥러닝 엔진 및 카운트 기반 NLP 시스템

---

## 1. Project Overview (프로젝트 개요)
본 프로젝트는 신경망의 내재적 수학 원리 이해와 행렬 연산 최적화 메커니즘을 규명하기 위해, 외부 딥러닝 프레임워크(PyTorch, TensorFlow 등) 의존성 없이 **Pure NumPy만을 활용하여 밑바닥부터 구축한 맞춤형 딥러닝 코어 엔진 및 자연어 처리(NLP) 임베딩 프로토타입**입니다.

대규모 데이터 연산 시 발생하는 성능 병목을 해결하기 위한 **Tensor-to-Matrix 고속 변환 최적화(`im2col`/`col2im`)**를 직접 구현하였으며, 복잡한 계층(Convolution, Batch Normalization 등)의 순전파 및 역전파 편미분 수식을 소스코드로 매핑하여 하드코딩 아키텍처의 구동 무결성을 증명했습니다.

---

## 2. Architectural Context (아키텍처 설계 의도)

![ARAI_Architecter_Diagram](https://github.com/junsu0302/ARIA/blob/main/imgs/ARIA%20Architecture%20Diagram.svg)

본 엔진의 핵심 연산 모듈 및 계층 모델들은 `core/` 패키지 내부에 독립적으로 격리되어 있습니다. 이는 다음과 같은 의도적 설계 프레임워크에 기반합니다.

* **결합도 완화 및 패키지 독립성**: 상위 애플리케이션 서빙 환경이나 테스트 스크립트와의 의존성을 완전히 차단하여, 연산 엔진 자체가 플러그인 형태로 어디서든 구동될 수 있도록 코어 레이어를 격리했습니다.
* **샌드박스 기반 구조적 실험**: 프레임워크 최적화 및 레이어 다각화 실험을 안정적으로 수행하기 위해 메인 루트와 연산 패키지를 분리하는 아키텍처 리팩토링을 적용했습니다.

---

## 3. Key Technical Challenges & Core Deep Dive (핵심 기술적 도전)

### 3.1 4차원 텐서 연산 최적화: `im2col` & `col2im` 파이프라인
합성곱 계층(Convolutional Layer) 구현 시 다중 중첩 루프(For-loop)로 인한 심각한 연산 병목을 해결하기 위해 고속 행렬 연산(GEMM, General Matrix Multiplication) 알고리즘을 도입했습니다.
* **im2col 변환**: 4차원 이미지 텐서 `(N, C, H, W)`에서 필터가 적용되는 영역을 추출하여 2차원 행렬로 펼쳐냄으로써, 하드웨어 레벨의 BLAS(Basic Linear Algebra Subprograms) 최적화 행렬 곱 연산이 가능하도록 아키텍처를 가속화했습니다.
* **col2im 변환**: 역전파(Backpropagation) 수행 시, 펼쳐진 2차원 오차 행렬을 원래의 4차원 텐서 공간의 기울기로 정확하게 복원 및 누적하는 전치 매핑 시스템을 구축했습니다.

### 3.2 완전 하드코딩 역전파와 수리적 체인 룰(Chain Rule) 통제
수학적 수식을 코드로 직접 매핑하여 프레임워크의 코어 연산 신뢰성을 확보했습니다.
* **Batch Normalization 계층 구현**: 학습 속도 개선과 미분 폭발(Exploding) 방지를 위해 평균과 분산을 이용한 정규화 순전파를 유도했습니다. 역전파 단계에서는 입력 데이터, 평균($\mu$), 분산($\sigma^2$), 감마($\gamma$), 베타($\beta$)에 대한 연쇄 법칙(Chain Rule) 편미분 수식을 완전히 분해하여 코드로 구현했습니다.
* **Convolution & Pooling 계층**: Spatial 차원 변화 속에서 국소적 기울기(Local Gradient)를 안전하게 역전파하고, Max Pooling의 `arg_max` 마스크 메커니즘을 추적하여 오차를 올바른 좌표로 전파하는 역전파 매커니즘을 완성했습니다.

---

## 4. Mathematical Sanity Check (수학적 무결성 검증)

본 엔진의 연산 모듈과 수식 전개가 완벽히 통제되고 있음을 입증하기 위해 투트랙 검증 시스템을 수행했습니다.

1. **수치 미분 기반 교차 검증 (Cross-Validation)**
   * `maths.py` 내부에 아주 작은 변화량($h = 10^{-4}$)과 중앙 차분(Central Difference) 공식을 사용하는 `numerical_gradient` 솔버를 구축했습니다.
   * 복잡한 레이어들의 해석적 역전파 기울기($dW$, $db$) 값과 수치 미분 값을 실시간 비교 검증하여, 두 값의 잔차 오차가 수리적 허용 오차 한계($10^{-7}$) 이하로 수렴함을 확인하여 수식 구현의 결함이 없음을 검증했습니다.
2. **MNIST 데이터셋 기반 엔드투엔드 학습 검증**
   * 구현된 `SimpleConvNet` 구조에 가중치 감쇄 및 Adam 옵티마이저를 결합하여 MNIST 정제 데이터셋에 대한 엔드투엔드 학습을 구동했습니다.
   * 검증 결과 **최고 테스트 정확도 약 99%**를 안정적으로 달성했습니다. 역전파 미분 수식의 부호가 반대로 매핑되거나 차원 누적 오류가 단 1개라도 발생할 경우 손실(Loss)이 즉시 발산하는 현상을 고려할 때, 본 성능 수치는 프레임워크 코어 엔진의 구조적·수학적 무결성을 증명하는 확정적 지표입니다.

---

## 5. Feature Specifications (주요 기능 명세)

| 분류 | 지원 모듈 및 구현 알고리즘 | 파일 위치 |
| :--- | :--- | :--- |
| **Layers** | Linear, Convolution, Pooling (Max), BatchNormalization, Dropout | `core/layers.py` |
| **Activations** | Sigmoid, ReLU, Softmax | `core/activations.py` |
| **Optimizers** | SGD, Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSprop, Adam | `core/optimizations.py` |
| **NLP Embedding** | Co-occurrence Matrix, Cosine Similarity, PPMI (양의 상호정보량), SVD (특이값 분해 차원축소) | `core/NLP.py` |

---

## 6. Future Engineering Roadmap (향후 아키텍처 확장 로드맵)

과거 수립했던 엔진 고도화 설계안을 기반으로 한 아키텍처 확장 로드맵입니다.

1. **동적 계산 그래프 및 자동 미분(Autograd) 엔진 탑재**
   * 각 레이어가 스스로 연산 그래프를 추적하고 `backward()`를 체이닝 호출하는 구조로 변환하여 PyTorch 스타일의 모듈화 수준 확보
2. **시계열 처리를 위한 Recurrent 계층 확장**
   * 시간축 3차원 데이터 처리를 위한 LSTM/GRU 계층 추가 및 BPTT(Backpropagation Through Time) 최적화 알고리즘 결합
3. **MLOps 파이프라인 연동을 위한 가중치 직렬화(Serialization)**
   * 학습 완료된 가중치와 편향 매개변수를 객체 형태로 직렬화하여 `.pkl` 또는 `.json` 파일로 Export/Import 하는 `save_weights()`, `load_weights()` 기능 고도화
