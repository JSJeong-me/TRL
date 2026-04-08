
# Module 7. PPO 실습: 보상모델과 가치모델이 들어가는 온라인 RL

## 1. 모듈 개요

이 모듈은 **오프라인 선호학습(DPO)** 다음 단계로, **온라인 RLHF 스타일의 PPO 루프**를 직접 경험하는 단계입니다.  
수강생은 같은 문제를 다시 보되, 이번에는 `prompt → generation → reward → PPO update`의 반복 구조로 바라보게 됩니다.

핵심 메시지는 다음과 같습니다.

- DPO는 `prompt / chosen / rejected`라는 **오프라인 선호쌍**으로 학습한다.
- PPO는 실제로 **응답을 생성한 뒤**, 그 응답에 대해 **reward를 계산하고**, 다시 정책(policy)과 가치(value)를 업데이트한다.
- 따라서 PPO는 DPO보다 **구성요소가 많고**, **실험 난이도도 더 높지만**, reward를 명시적으로 정의할 수 있는 과제에서는 더 직접적인 최적화가 가능하다.

이번 모듈에서는 복잡한 human preference 데이터 대신, **규칙 기반 reward(rule-based reward)** 로 시작합니다.  
예를 들면 다음과 같습니다.

- 수학 과제: 정답 일치 시 높은 점수
- JSON 과제: parse 가능하고 필요한 key가 있으면 높은 점수
- persona 과제: 공손한 표현과 길이 제약을 만족하면 가점
- safety 과제: 직접 위험 조언을 하지 않으면 가점

---

## 2. 학습 목표

이 모듈이 끝나면 수강생은 다음을 설명할 수 있어야 합니다.

1. PPO가 왜 “온라인 RLHF형 루프”인지 설명할 수 있다.
2. reward model, value model, KL penalty가 왜 등장하는지 개념적으로 설명할 수 있다.
3. rule-based reward로도 PPO 실험을 시작할 수 있음을 이해한다.
4. `objective/kl`, `objective/scores`, `loss/value_avg` 같은 로그를 읽을 수 있다.
5. 같은 평가셋에서 **SFT vs PPO**를 비교하고, PPO가 잘한 점과 어려운 점을 정리할 수 있다.

---

## 3. 강의 흐름

### Part A. 개념 설명
- PPO는 policy gradient 계열 방법이다.
- 샘플을 수집하고, surrogate objective로 여러 minibatch epoch를 수행한다.
- RLHF 맥락에서는 “생성 → 보상 → 정책 업데이트” 구조를 가진다.
- value function은 reward 예측/advantage 계산의 일부로 이해한다.
- KL 항은 정책이 너무 급격히 바뀌지 않게 잡아주는 역할로 설명한다.

### Part B. DPO와의 비교
- DPO: 오프라인 preference pair
- PPO: 온라인 generation + reward + value + KL
- DPO는 간단하고 가볍다.
- PPO는 복잡하지만 reward를 직접 정의할 수 있다.

### Part C. reward 설계 실습
- 같은 `module3_ppo_dataset.jsonl`을 읽는다.
- task_type별 reward 함수를 만든다.
- reward가 무엇을 보상하고 무엇을 벌점하는지 토론한다.

### Part D. PPO 실행
- SFT 모델을 출발점으로 사용한다.
- query를 생성하고 reward를 계산한다.
- PPO step을 반복한다.
- 로그를 저장하고 지표를 해석한다.

### Part E. 전후 비교
- 같은 evaluation prompt 세트에 대해 SFT 출력과 PPO 출력 비교
- 스타일, 정답성, 형식 준수, 안전성 비교
- PPO가 잘하는 항목 / 못하는 항목 정리

---

## 4. 슬라이드 구성안

### Slide 1. 제목
**Module 7. PPO 실습: 보상모델과 가치모델이 들어가는 온라인 RL**

### Slide 2. 왜 PPO인가
- DPO로는 “선호 비교”를 잘 배울 수 있다.
- 하지만 reward가 명확한 과제에서는 PPO가 더 직접적이다.
- 예: exact answer, strict format, constraint satisfaction

### Slide 3. PPO의 한 줄 정의
- `sample → score → optimize surrogate objective`

### Slide 4. RLHF에서의 PPO 루프
- query 입력
- model response 생성
- reward model 또는 rule-based reward 계산
- policy/value update
- KL penalty로 drift 제어

### Slide 5. 이번 실습의 보상 설계
- math: exact match
- format: JSON parse + required keys
- persona: polite + concise
- safety: safe refusal

### Slide 6. PPO 로그 읽기
- `objective/scores`
- `objective/kl`
- `objective/rlhf_reward`
- `loss/policy_avg`
- `loss/value_avg`
- `val/ratio`

### Slide 7. DPO vs PPO 비교
- DPO는 offline pair
- PPO는 online rollout
- DPO는 단순/경량
- PPO는 직접 reward 최적화 가능

### Slide 8. 실습 결과 해석
- score가 오르는가
- KL이 과도하게 커지는가
- value loss가 튀는가
- 특정 task에서만 개선되는가

---

## 5. 실습 구성

### 사용 입력 파일
- `module3_ppo_dataset.jsonl`
- `module4_sft_output/` (또는 base model fallback)

### 생성 산출물
- `module7_ppo_training_stats.jsonl`
- `module7_ppo_eval_comparison.json`
- `module7_ppo_observation.md`

### 핵심 실습 순서
1. PPO source dataset 불러오기
2. reward function 구현
3. query tokenization
4. SFT model + value head로 PPO 시작
5. training stats 저장
6. SFT vs PPO 비교 평가

---

## 6. 실습 중 강사가 강조할 포인트

### 6-1. PPO는 정답 예시를 직접 따라 하는 방식이 아니다
SFT와 달리 PPO는 정답 텍스트를 직접 복사해서 학습하는 것이 아니라,
현재 정책이 생성한 응답을 reward로 평가한 뒤 그 reward가 커지는 쪽으로 policy를 조정한다.

### 6-2. reward 설계가 곧 목표 설계다
reward가 잘못되면 모델은 엉뚱한 방향으로 최적화된다.
예를 들어 JSON parse만 보상하면 의미 없는 JSON도 높은 점수를 받을 수 있다.
그래서 `parse 가능` + `required keys 존재` 같이 다중 조건이 필요하다.

### 6-3. KL과 value loss를 꼭 같이 보자
- `objective/scores`만 보면 reward는 오르는데
- `objective/kl`가 너무 크면 정책이 급격히 흔들릴 수 있다.
- `loss/value_avg`가 불안정하면 value prediction이 잘 안 되는 신호일 수 있다.

### 6-4. PPO는 DPO보다 무겁다
이 실습은 교육용이므로 아주 작은 데이터셋과 짧은 step으로 한다.
핵심은 성능 최대화가 아니라 **루프 구조와 로그 해석**이다.

---

## 7. 과제 템플릿

### 과제 제목
**Rule-based reward를 사용해 PPO를 수행하고 SFT vs PPO를 비교하라**

### 제출물
1. `module7_ppo_training_stats.jsonl`
2. `module7_ppo_eval_comparison.json`
3. `module7_ppo_observation.md`

### 제출 질문
- 어떤 reward 규칙이 가장 잘 작동했는가?
- PPO 이후 어떤 category가 가장 좋아졌는가?
- `objective/kl`와 `loss/value_avg`는 안정적이었는가?
- 이 문제는 PPO가 DPO보다 유리했는가?
- reward를 어떻게 개선하면 더 좋아질 것 같은가?

---

## 8. 평가 기준

### 우수
- reward 함수가 구체적이고 일관적이다.
- 로그를 읽고 PPO 동작을 설명할 수 있다.
- SFT vs PPO 차이를 category별로 잘 정리했다.
- PPO의 장점과 한계를 모두 언급했다.

### 보통
- reward는 동작하지만 설명이 약하다.
- 일부 로그만 해석했다.
- 개선/악화 사례를 제한적으로 언급했다.

### 미흡
- reward 함수가 task와 맞지 않는다.
- 로그 의미를 해석하지 못한다.
- DPO와 PPO 차이를 설명하지 못한다.

---

## 9. 마무리 멘트

이 모듈의 핵심은 “PPO가 더 좋다”가 아닙니다.  
핵심은 **DPO와 PPO가 문제를 바라보는 방식이 다르다**는 것을 체감하는 것입니다.

- DPO는 “둘 중 어느 답이 더 낫나?”
- PPO는 “이 응답에 몇 점을 줄 것인가?”

이 차이를 이해하면, 다음에 reward model을 붙이거나 더 복잡한 RLHF 실험으로 확장할 때 훨씬 자연스럽게 넘어갈 수 있습니다.
