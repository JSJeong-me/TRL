# Module 8. GRPO 실습: PPO와 무엇이 같고 무엇이 다른가

## 1. 모듈 개요

이 모듈은 **GRPO(Group Relative Policy Optimization)** 를 독립적으로 실습하면서, 바로 앞 Module 7의 PPO 실습과 정면 비교하도록 설계합니다.  
DeepSeekMath는 GRPO를 **PPO의 변형**으로 소개하면서, 수학 추론 성능을 강화하면서도 **PPO의 메모리 사용을 줄이는 방향**을 강조합니다. Hugging Face TRL의 GRPO 문서도 GRPO를 **prompt 중심 + reward function 중심**으로 빠르게 실험할 수 있게 안내하며, GRPO는 온라인 학습 알고리즘으로서 모델이 생성한 completion에 reward를 부여하고, 그룹 내 상대 비교 기반 advantage를 사용해 업데이트한다고 설명합니다. 또한 현재 TRL 문서는 **기본적으로 `beta=0.0`** 으로 설정되어 KL 항을 사용하지 않는 구성을 기본값으로 둡니다.  
참고: DeepSeekMath(https://arxiv.org/abs/2402.03300), TRL GRPO Trainer(https://huggingface.co/docs/trl/grpo_trainer)

이번 실습은 PPO와 가능한 한 **동일 조건**으로 맞춥니다.

- 같은 초기 모델: `module4_sft_output`
- 같은 또는 유사한 prompt 세트: `module3_grpo_dataset.jsonl`
- 같은 보상 철학: rule-based reward
- 같은 비교 질문:
  - 학습 안정성은 어떤가?
  - GPU 메모리 사용량은 어떤가?
  - 구현 난이도는 어떤가?
  - 응답 품질은 어떤가?
  - 보상 상승 속도는 어떤가?

이 모듈의 목적은 **“GRPO가 PPO보다 무조건 낫다”** 를 보여주는 것이 아니라,  
**critic/value model 의존성이 줄어든 온라인 RL 실험이 어떤 실무적 장단점을 갖는지** 체감하게 하는 데 있습니다.

---

## 2. 학습 목표

모듈 종료 시 수강생은 다음을 설명할 수 있어야 합니다.

1. **DPO는 offline preference alignment** 라고 설명할 수 있다.
2. **PPO는 reward/value 기반 online RL** 이라고 설명할 수 있다.
3. **GRPO는 PPO 계열이지만 그룹 상대 비교를 활용하며, critic/value-model 없이 더 경량화된 RL 실험을 구성할 수 있다**고 설명할 수 있다.
4. 동일한 reward 함수라도 PPO와 GRPO에서 **운영 복잡도와 메모리 사용**이 다르게 느껴질 수 있음을 설명할 수 있다.
5. 동일한 prompt 세트와 reward 기준으로 **SFT vs PPO vs GRPO**를 비교 기록할 수 있다.

---

## 3. 핵심 개념 설명

### 3-1. GRPO는 왜 PPO의 변형인가

DeepSeekMath는 GRPO를 **PPO의 변형**으로 설명하며, 특히 **critic model을 두지 않고 그룹 점수로 baseline을 추정**하여 학습 자원을 줄인다고 소개합니다.  
즉 PPO가 보통 policy + value 구조를 함께 다루는 반면, GRPO는 생성된 여러 completion의 **상대적 reward**를 이용해 advantage를 만들고 업데이트합니다.

### 3-2. TRL 문서에서 보는 GRPO의 특징

Hugging Face TRL 문서는 GRPO를 다음처럼 설명합니다.

- GRPO는 **online learning algorithm**
- 각 step에서 prompt 배치를 뽑고, prompt당 여러 completion을 생성
- reward model 또는 reward function으로 각 completion을 평가
- 그룹 내 평균/표준편차를 기준으로 상대 advantage 계산
- reference policy와의 거리도 고려할 수 있으나, **현재 문서상 기본 `beta=0.0`** 으로 KL 항은 기본 비활성

즉, 교육적으로는 다음처럼 요약할 수 있습니다.

- PPO: reward + value + KL 기반 온라인 RL
- GRPO: reward + group-relative advantage + optional KL 기반 온라인 RL

### 3-3. 이번 모듈에서 사용할 비교 프레임

이번 실습은 PPO와 최대한 공정하게 맞춥니다.

- task_type은 동일: math / format / persona / safety
- reward 함수 철학은 동일: exact match, JSON parse, polite+brief, safe refusal
- 초기 모델은 동일: Module 4 SFT 결과
- 비교 지표:
  - reward 평균
  - 생성 품질
  - peak GPU memory
  - step time
  - 구현 체감 난이도

---

## 4. 강의 슬라이드 구성안

### Slide 1. 제목
**Module 8. GRPO 실습: PPO와 무엇이 같고 무엇이 다른가**

### Slide 2. 복습
- DPO = offline preference alignment
- PPO = reward/value 기반 online RL
- GRPO = PPO 변형, group-relative advantage

### Slide 3. DeepSeekMath의 메시지
- GRPO는 PPO variant
- critic을 생략해 메모리 사용 감소 방향
- 수학/추론 계열 task에 특히 잘 알려짐

### Slide 4. TRL 기준 GRPO 흐름
- prompt 샘플링
- prompt당 여러 completion 생성
- reward 계산
- 그룹 상대 advantage 계산
- policy update

### Slide 5. PPO와 같은 점
- 둘 다 online RL
- 둘 다 rollout이 필요
- 둘 다 reward function 또는 reward model 필요
- 둘 다 reference policy / KL 개념을 가질 수 있음

### Slide 6. PPO와 다른 점
- PPO는 value/critic 구조가 핵심
- GRPO는 critic 없이 group-relative baseline 사용
- GRPO는 실습 구성상 더 가볍게 느껴질 수 있음
- GRPO는 동일 prompt에 여러 completion을 두는 사고가 특히 중요

### Slide 7. 이번 실습 구조
- SFT 초기 모델 로드
- `module3_grpo_dataset.jsonl` 로드
- reward 함수 정의
- GRPOTrainer 실행
- before / after 비교
- PPO 결과와 비교 메모 작성

### Slide 8. 비교 질문
- reward 상승 속도는 어떤가?
- 학습 안정성은 어떤가?
- GPU 메모리는 어느 쪽이 더 부담인가?
- 구현 난이도는 어느 쪽이 더 높게 느껴지는가?
- 최종 출력 품질은 어떤 차이를 보이는가?

---

## 5. 실습 구성

### 5-1. 입력 파일
- `/content/module3_grpo_dataset.jsonl`
- `/content/module4_sft_output` (또는 같은 역할의 SFT 모델 경로)

### 5-2. reward 함수
PPO에서 썼던 철학을 그대로 가져옵니다.

- `math_reward_func`: 정답 일치
- `format_reward_func`: JSON parse + required_keys
- `persona_reward_func`: 공손한 표현 + 길이 제한
- `safety_reward_func`: 위험 요청에 대한 안전한 거절

### 5-3. 출력 파일
- `module8_grpo_before_after.json`
- `module8_grpo_scorecard.json`
- `module8_grpo_summary.md`

---

## 6. 강사용 설명 포인트

### 포인트 1
GRPO는 “reward가 있으면 되는 PPO-lite”가 아니라,  
**같은 prompt에 대해 여러 completion을 뽑고 그 상대적 차이를 이용하는 사고방식**이 중요합니다.

### 포인트 2
이번 실습에서는 PPO와 동일 reward 철학을 사용하므로,  
변수는 최대한 **학습 알고리즘 차이** 쪽에 모이게 설계합니다.

### 포인트 3
GRPO는 value model이 없다고 해서 자동으로 항상 더 낫다는 뜻은 아닙니다.  
오히려 어떤 task에서는 PPO의 구조가 더 해석하기 쉬울 수 있습니다.  
이번 실습은 정답 비교보다 **trade-off 이해**가 목적입니다.

---

## 7. 실습 과제

### 과제 제목
**같은 reward 체계에서 PPO와 GRPO를 비교하라**

### 제출물
1. GRPO 학습 로그
2. SFT vs GRPO 출력 비교 4건
3. PPO vs GRPO 비교 메모
4. `module8_grpo_summary.md`

### 메모 질문
- GRPO는 PPO보다 메모리 부담이 실제로 줄어들었는가?
- reward가 비슷한 속도로 상승했는가?
- 어떤 task에서 GRPO가 더 잘 맞았는가?
- 어떤 부분은 PPO가 더 해석하기 쉬웠는가?
- 실무에서 두 방법 중 무엇을 먼저 시도하겠는가?

---

## 8. 다음 단계 연결

이 모듈이 끝나면 다음 종합 단계로 아래를 연결할 수 있습니다.

- SFT vs DPO vs PPO vs GRPO 통합 비교
- 과제별 최적 학습법 선택
- “선호 정렬 문제”와 “명시적 reward 문제” 분리
- 개인화 / 형식 준수 / 추론 강화에 대한 알고리즘 선택 전략
