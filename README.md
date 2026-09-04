# 🏠 주택 가격 예측

> 머신러닝을 활용하여 주택 가격을 예측하고,
> 주택 가격에 영향을 미치는 주요 요인을 분석한 프로젝트입니다.

---

## 1. Project Overview

### 프로젝트 목적
주택의 다양한 특성을 활용하여 가격을 예측하고,
각 변수들이 주택 가격에 어떤 영향을 미치는지 분석합니다.

### 주요 목표
- 주택 가격에 영향을 미치는 주요 변수 파악
- 데이터 전처리 및 Feature Engineering
- 여러 회귀 모델의 성능 비교
- 최종 모델 선정 및 결과 해석

### 담당 역할

- 데이터 특성을 고려한 결측치 처리 및 `Lot Frontage` 예측 기반 결측값 대치
- Ridge·Lasso·ElasticNet 등 회귀 모델 성능 비교 및 최종 모델 선정
- 모델 성능 및 예측 결과 시각화
- 분석 결과 보고서 작성 및 최종 발표
- 팀원별 역할 분담 및 회의 내용 정리를 통한 프로젝트 진행 관리

---

## 2. Problem Definition

본 프로젝트에서는 다음 질문에 답하고자 했습니다.

1. 어떤 변수가 주택 가격에 가장 큰 영향을 미치는가?
2. 범주형 변수를 어떻게 처리하는 것이 효과적인가?
3. 어떤 머신러닝 모델이 가장 높은 예측 성능을 보이는가?

---

## 3. Dataset

- 학습 데이터: `regression_train_public.csv`
- 예측 데이터: `regression_test_private.csv`
- Target Variable: `SalePrice`
- 학습 데이터 크기: 약 2,344개 관측치
- 분석 변수: 30개 설명변수 + 1개 Target 변수

주택의 면적, 건축연도, 주거 품질, 지역, 차고, 지하실 등
다양한 주택 특성 정보를 활용하여 판매가격을 예측했습니다.

### 주요 변수 예시

- `Overall Qual` : 주택의 전반적인 품질
- `Gr Liv Area` : 지상 거주 면적
- `Lot Area` : 대지 면적
- `Neighborhood` : 주택 위치 지역
- `Year Built` : 건축 연도
- `Garage Cars` : 차고 수용 차량 수
- `Total Bsmt SF` : 지하실 면적
- `Kitchen Qual` : 주방 품질
  
---

## 4. Analysis Process

Data Collection  
↓  
EDA  
↓  
Missing Value Handling  
↓  
Outlier Handling  
↓  
Feature Engineering  
↓  
Categorical Encoding  
↓  
Feature Scaling  
↓  
Feature Selection  
↓  
Modeling  
↓  
Model Evaluation  
↓  
Insight

---

## 5. Data Preprocessing

### 5.1 Missing Value Handling

단순 평균 대치보다는 변수의 의미와 다른 변수와의 관계를 고려하여
결측치를 처리했습니다.

- Garage 관련 변수  
  → 차고가 없는 경우 `NoGarage`로 처리
- Basement 관련 변수  
  → 지하실이 없는 경우 `NoBsmt`로 처리
- Fireplace 관련 변수  
  → 벽난로가 없는 경우 `NoFireplace`로 처리
- `Lot Frontage`  
  → 다른 주택 특성을 활용한 별도의 회귀 모델을 학습하여 결측값 예측

특히 `Lot Frontage`의 경우 Ridge, Lasso, ElasticNet을 비교하였으며,
CV RMSE가 가장 낮은 Lasso 모델을 이용하여 결측값을 대치했습니다.

### 5.2 Log Transformation

면적 변수처럼 극단적으로 큰 값이 존재하는 변수의 분포를 완화하고, 극단값이 회귀 모델에 과도한 영향을 주는 것을 줄이기 위해 log1p 변환을 적용했습니다.

대상 변수:

- `Lot Area`
- `Open Porch SF`
- `Lot Frontage`
- `Total Bsmt SF`
- `Gr Liv Area`

이를 통해 극단적으로 치우친 분포를 완화하고
회귀 모델의 안정성을 높이고자 했습니다.

### 5.3 Outlier Handling

로그 변환 이후에도 극단값이 존재하는 변수에 대해
IQR 기반 이상치 처리를 적용했습니다.

대상 변수:

- `Lot Area`
- `Gr Liv Area`
- `Total Bsmt SF`
- `Open Porch SF`
- `Lot Frontage`

### 5.4 Feature Scaling

변수 간 단위 차이에 따른 회귀 계수 왜곡을 줄이기 위해
`RobustScaler`를 적용했습니다.

이상치의 영향을 상대적으로 적게 받는 RobustScaler를 사용하여
연속형 변수의 스케일을 조정했습니다.

---

## 6. Modeling

다음 정규화 기반 회귀 모델을 비교했습니다.

### Ridge Regression

L2 규제를 적용하여 계수의 크기를 축소하고
다중공선성 문제를 완화했습니다.

상관계수와 Mutual Information을 활용하여
변수 선택 기준을 달리한 3개의 Ridge 모델을 구성했습니다.

### Lasso Regression

L1 규제를 이용하여 중요도가 낮은 변수의 계수를 0으로 만들어
자동으로 변수 선택을 수행했습니다.

- 최적 Alpha: `100`
- 선택 변수: 54개

### ElasticNet

L1과 L2 규제를 동시에 적용하여
변수 선택과 모델 안정성을 함께 확보했습니다.

- 최적 Alpha: `0.01`
- L1 Ratio: `0.5`
  
하이퍼파라미터는 `GridSearchCV`와 5-Fold Cross Validation을 이용하여
선정했습니다.

---

## 7. Results

### 모델 성능 비교

| Model | CV RMSE | Test RMSE |
|---|---:|---:|
| ElasticNet | 24,415.61 | **20,681.72** |
| Lasso | 24,472.02 | 20,970.32 |
| Ridge 3차 | 24,527.24 | 20,917.05 |
| Ridge 1차 | 24,671.98 | 21,348.17 |
| Ridge 2차 | 25,978.59 | 22,451.37 |

가장 낮은 Test RMSE를 기록한 **ElasticNet을 최종 모델로 선정**했습니다.

### Final Model Performance

- Test R²: **0.9152**
- Test Adjusted R²: **0.8941**
- Test MAPE: **9.28%**
- Test RMSE: **20,681.72**

Train과 Test 간 성능 차이가 크지 않아
과적합 위험이 비교적 낮고 일반화 성능이 안정적인 것으로 판단했습니다.

---

## 8. Key Challenge & Solution

### 8.1 Lot Frontage 결측치 처리
단순 평균 대치 대신 Ridge, Lasso, ElasticNet 모델을 비교하여
가장 성능이 좋은 Lasso 기반 예측값으로 결측치를 대치했습니다.

### 8.2 범주형 변수 인코딩 전략
범주형 변수의 특성을 고려하여
Target Encoding과 One-Hot Encoding을 선택적으로 적용했습니다.

---

## 9. Key Insights

### 1. 주택 가격은 단일 변수보다 여러 특성이 복합적으로 작용한다

주거 면적, 주택 품질, 지역, 지하실, 차고 등의 변수가
주택 가격 예측에 함께 영향을 미쳤습니다.

### 2. 주거 면적과 주택 품질이 중요한 가격 결정 요인으로 나타났다

최종 모델과 Ridge/Lasso 분석에서

- `Gr Liv Area`
- `Overall Qual`
- `Total Bsmt SF`

등의 변수가 높은 영향력을 보였습니다.

### 3. 지역(Neighborhood)에 따른 가격 차이가 크게 나타났다

`StoneBr`, `NoRidge`, `NridgHt` 등 일부 지역 변수의 회귀계수가
상대적으로 크게 나타나 주택의 위치가 가격 형성에 중요한 요소임을 확인했습니다.

### 4. 단순한 변수 제거보다 규제 기반 모델이 효과적이었다

Ridge, Lasso, ElasticNet을 비교한 결과
L1과 L2 규제를 함께 사용하는 ElasticNet이 가장 낮은 Test RMSE를 기록했습니다.

이는 다양한 주택 특성 변수들이 서로 연관되어 있는 데이터에서
ElasticNet이 변수 선택과 계수 안정성을 동시에 확보하는 데 효과적이었음을 보여줍니다.

---

## 10. Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`
