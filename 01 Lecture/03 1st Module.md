아래는 **1st Module**의 상세 설계안입니다. 이 모듈은 뒤의 SFT·DPO·PPO·GRPO 실습 전체를 이해하기 위한 **개념 앵커(anchor)** 역할을 하도록 설계했습니다. DPO는 선호쌍 기반의 직접 최적화, PPO는 온라인 샘플링과 surrogate objective를 쓰는 정책경사 계열, GRPO는 DeepSeekMath에서 제시된 PPO 변형으로 수학 추론 강화와 메모리 절감을 강조하며, 현재 TRL은 SFT·DPO·PPO·GRPO 계열 실습 진입점을 문서와 Trainer로 제공합니다. ([Hugging Face][1])

# Module 1

**Post-Training 전체 지도와 PPO·DPO·GRPO 비교 입문**

## 1. 모듈 개요

이 모듈의 목적은 “기법을 외우는 것”이 아니라, 수강생이 **같은 문제를 보고도 어떤 post-training 방법을 선택해야 하는지 판단하는 기준**을 갖게 만드는 것입니다.
특히 이 모듈에서는 다음 세 가지를 분명히 잡습니다.

* **SFT**는 기본 능력, 말투, 형식, 지시 따르기 성향을 먼저 안정화하는 출발점
* **DPO**는 `prompt / chosen / rejected` 형태의 선호쌍 데이터를 사용해 직접 선호 정렬을 수행하는 방식
* **PPO / GRPO**는 보상 신호를 이용하는 RL 계열이며, PPO는 온라인 RL의 대표적 방식이고 GRPO는 PPO 계열의 경량 변형으로 다룹니다. ([arXiv][2])

### 권장 시간

총 2.5시간 기준으로 설계합니다.

* 강의 70분
* 데모 20분
* 실습 40분
* 토론 및 정리 20분

### 선수 지식

Python 기초, Hugging Face 개념을 아주 조금 알고 있으면 좋지만, 이 모듈은 아직 본격 튜닝을 하지 않으므로 필수는 아닙니다.

### 모듈 산출물

이 모듈이 끝나면 수강생은 아래 3가지를 제출할 수 있어야 합니다.

1. **SFT / DPO / PPO / GRPO 비교표**
2. **같은 과제를 3가지 데이터 형식으로 바꾼 예시**
3. **어떤 상황에서 어떤 기법을 선택할지에 대한 1페이지 메모**

---

## 2. 학습 목표

수강생은 모듈 종료 시 다음을 설명할 수 있어야 합니다.

* pretraining과 post-training의 역할 차이
* 왜 SFT만으로는 한계가 생기고, 왜 preference learning 또는 RL이 필요한지
* DPO와 PPO의 가장 큰 차이
* PPO와 GRPO의 공통점과 차이
* 동일한 문제를 SFT용 데이터, DPO용 데이터, RL용 보상 문제로 각각 바꾸는 방법

이 모듈의 학습 목표는 TRL 문서가 제시하는 post-training 흐름과, DPO·PPO·GRPO 원전/대표 논문의 핵심 차이를 교육용 수준으로 재구성한 것입니다. ([Hugging Face][1])

---

## 3. 교안 초안

## 3-1. 슬라이드 구성안

### Slide 1. 강의 제목

**“Model Post-Training의 전체 지도: SFT, DPO, PPO, GRPO”**

강사 멘트:

> 오늘의 목표는 정의 암기가 아니라, “어떤 문제에 어떤 방법이 맞는가”를 판단하는 기준을 만드는 것입니다.

---

### Slide 2. 왜 Post-Training이 필요한가

내용:

* Pretraining은 일반 언어 능력을 형성
* 하지만 원하는 말투, 정체성, 안전성, 선호, 특정 능력은 별도 조정이 필요
* 그래서 instruction tuning, preference alignment, RL 기반 개선이 등장

강사 설명 포인트:

* “잘 말하는 모델”과 “내가 원하는 방식으로 답하는 모델”은 다르다
* Post-training은 후자에 가깝다

---

### Slide 3. 전체 흐름 지도

내용:
**Base model → SFT → DPO 또는 PPO/GRPO → 평가 → 개선 반복**

강조:

* SFT는 대개 공통 출발점
* DPO는 오프라인 preference 데이터 중심
* PPO/GRPO는 reward 중심

이 흐름은 TRL quickstart가 SFT, DPO, GRPO를 대표 경로로 제시하는 구조와 잘 맞습니다. ([Hugging Face][1])

---

### Slide 4. SFT 한 장 요약

내용:

* 데이터: instruction → response
* 목적: 형식, 말투, 기본 작업 능력 맞추기
* 장점: 단순하고 안정적
* 한계: “둘 중 더 나은 답”의 상대 선호를 직접 학습하긴 어려움

강사 설명:

* SFT는 “정답 예시를 보여주는 교육”
* 하지만 “이 둘 중 어느 답을 더 좋아하는가”는 다른 문제

---

### Slide 5. DPO 한 장 요약

내용:

* 데이터: `prompt, chosen, rejected`
* 목적: 더 선호되는 답을 직접 학습
* 특징: 별도 reward model 없이 preference alignment 수행
* 장점: RLHF보다 단순
* 한계: 좋은 preference pair 품질이 핵심

DPO 논문은 DPO를 “explicit reward model이나 reinforcement learning 없이도” 선호를 직접 최적화하는 방법으로 제시합니다. ([arXiv][2])

---

### Slide 6. PPO 한 장 요약

내용:

* 데이터: prompt + generation + reward
* 목적: 보상을 크게 만드는 방향으로 정책 개선
* 특징: rollout, reward, update 반복
* 장점: 명시적 reward 최적화 가능
* 한계: 구성 복잡도와 계산비용이 큼

PPO는 상호작용으로 데이터를 수집하고 surrogate objective로 여러 minibatch epoch를 수행하는 정책경사 계열 방법입니다. ([arXiv][3])

---

### Slide 7. GRPO 한 장 요약

내용:

* PPO 계열의 변형
* 그룹 상대 비교 기반 사고
* reasoning/수학형 과제에서 많이 언급
* PPO 대비 더 가볍게 실험하려는 맥락에서 소개 가능

DeepSeekMath는 GRPO를 PPO의 변형으로 설명하며, 수학 추론 성능 향상과 메모리 사용 절감을 강조합니다. TRL 역시 GRPOTrainer와 reward 함수를 이용한 quick example을 제공합니다. ([arXiv][4])

---

### Slide 8. 네 방법 비교표

아래 비교표는 원논문과 TRL 문서를 바탕으로 강의용으로 단순화한 것입니다. ([arXiv][2])

| 항목                 | SFT            | DPO       | PPO             | GRPO                   |
| ------------------ | -------------- | --------- | --------------- | ---------------------- |
| 핵심 데이터             | 정답 예시          | 선호쌍       | 보상 가능한 출력       | 보상 가능한 출력              |
| 학습 형태              | 지도학습           | 오프라인 선호학습 | 온라인 RL          | RL 계열                  |
| reward model 필요    | 보통 없음          | 없음        | 있을 수 있음         | 보통 reward 함수 중심 실습 가능  |
| value model/critic | 없음             | 없음        | 일반적으로 사용        | PPO 대비 단순화된 변형으로 교육 가능 |
| 강의 난이도             | 가장 쉬움          | 중간        | 높음              | 중간~높음                  |
| 잘 맞는 과제            | 스타일, 형식, 기본 능력 | 선호 정렬     | 보상 정의 가능한 능력 향상 | reasoning/정답 기반 과제 비교  |

강사 멘트:

> 오늘은 이 표를 암기하는 것이 아니라, 뒤의 실습에서 직접 채워 나갈 기준표로 사용합니다.

---

### Slide 9. “같은 문제”를 다른 방식으로 바꾸기

예시 문제:
“모델이 더 친절하고 짧게 답하게 만들고 싶다.”

이를 바꾸는 방식:

* SFT: 친절하고 짧은 정답 예시를 많이 준다
* DPO: 친절·짧은 답 vs 장황·무례한 답의 선호쌍을 만든다
* PPO: 친절성/길이/형식 준수에 대한 reward를 만든다
* GRPO: 같은 reward 문제를 그룹 상대 비교형 실험으로 다룬다

이 슬라이드가 Module 1의 핵심입니다.

---

### Slide 10. 실무에서 어떤 방법을 먼저 고를까

내용:

* 데이터만 충분하면 SFT부터
* preference pair가 이미 있으면 DPO 유리
* 정답 검증 또는 rule-based reward가 쉬우면 PPO/GRPO 고려
* 계산 자원이 약하면 DPO 쪽이 교육용으로 더 부담이 적음

DPO 논문은 RLHF 대비 단순성과 경량성을 장점으로 내세우고, PPO는 본질적으로 rollout과 policy optimization이 들어가므로 실습 구성이 더 복잡합니다. ([arXiv][2])

---

### Slide 11. 오늘 실습 안내

내용:
같은 5개의 질문을 보고

1. SFT용 정답 예시 작성
2. DPO용 chosen/rejected 작성
3. PPO/GRPO용 reward 규칙 정의

즉, 아직 튜닝하지 않고 **데이터/문제 설계 감각**을 먼저 익힙니다.

---

### Slide 12. 정리

핵심 문장:

* SFT는 “정답을 보여주는 방식”
* DPO는 “더 나은 답을 고르게 하는 방식”
* PPO는 “보상을 크게 만드는 방향으로 행동을 고치는 방식”
* GRPO는 “PPO 계열을 더 실용적으로 쓰려는 변형으로 이해”

---

## 3-2. 강의 진행 시나리오

### Part A. 개념 설명 20분

강사는 먼저 pretraining과 post-training을 구분하고, post-training 안에서도

* example imitation
* preference alignment
* reward optimization
  이라는 세 축이 있음을 설명합니다.

### Part B. 비교 프레임 제시 20분

수강생에게 무조건 아래 4개 질문으로 생각하게 합니다.

1. 데이터는 무엇이 필요한가
2. 학습은 offline인가 online인가
3. 구현은 얼마나 복잡한가
4. 무엇을 가장 잘 바꿀 수 있는가

### Part C. 사례 설명 15분

같은 질문을 예시로 들고 SFT, DPO, PPO, GRPO로 각각 재정의합니다.

### Part D. 실습 전 데모 15분

강사가 한 문제를 즉석에서 세 방식으로 바꿔 보여줍니다.

예:
“모델이 한국어로 더 공손하게 답하게 만들고 싶다.”

* SFT 데이터 한 건 작성
* DPO pair 한 건 작성
* reward 함수 초안 한 건 작성

---

## 4. 실습 예제 코드 주제

이 Module 1은 아직 본격 학습 코드를 길게 작성하는 단계가 아닙니다.
대신 뒤 모듈에서 재사용할 **준비용 코드 주제**를 제공합니다.

## 4-1. 코드 주제 A

**`compare_training_formats.ipynb`**

목적:
하나의 raw task를 받아서

* SFT 포맷
* DPO 포맷
* PPO/GRPO용 reward task 포맷
  으로 자동 변환해 보는 노트북

핵심 기능:

* 입력: `task`, `good_answer`, `bad_answer`
* 출력:

  * SFT JSON
  * DPO JSON
  * RL prompt + reward rubric JSON

예시 출력 구조:

```json
{
  "sft": {
    "prompt": "Explain this politely in Korean.",
    "response": "안녕하세요. 아래와 같이 설명드리겠습니다..."
  },
  "dpo": {
    "prompt": "Explain this politely in Korean.",
    "chosen": "안녕하세요. 아래와 같이 설명드리겠습니다...",
    "rejected": "그래서 답은 이거예요."
  },
  "rl": {
    "prompt": "Explain this politely in Korean.",
    "reward_rules": [
      "Korean language used",
      "contains greeting",
      "less than 120 characters"
    ]
  }
}
```

---

## 4-2. 코드 주제 B

**`preference_pair_builder.py`**

목적:
주어진 질문과 여러 개의 답변 후보를 입력받아 DPO용 `chosen/rejected` 쌍을 만드는 연습용 스크립트

핵심 포인트:

* preference 기준을 코드상 파라미터로 둠
* 예: `politeness`, `brevity`, `correctness`, `format_compliance`
* 아직 자동 라벨링 정확도가 핵심은 아니고, **preference라는 것이 결국 기준 의존적**이라는 점을 체감하게 하는 것이 목적

---

## 4-3. 코드 주제 C

**`reward_sandbox.py`**

목적:
PPO/GRPO용 reward 설계를 감각적으로 익히는 샌드박스

핵심 기능:

* 문자열 길이
* 특정 포맷 준수 여부
* 정답 일치 여부
* 금지 표현 포함 여부

같은 출력에 대해 reward가 어떻게 달라지는지 확인합니다.

예시:

```python
def reward_fn(answer: str) -> float:
    score = 0.0
    if answer.startswith("안녕하세요"):
        score += 0.3
    if len(answer) < 120:
        score += 0.3
    if "죄송" not in answer:
        score += 0.2
    if "JSON" in answer:
        score += 0.2
    return score
```

이 코드는 PPO/GRPO 실습 이전에 “reward가 곧 목표 설계”라는 점을 이해시키는 데 유용합니다.

---

## 5. Module 1 실습 과제 템플릿

## 5-1. 과제 제목

**같은 문제를 SFT / DPO / PPO(GRPO) 문제로 재구성하기**

## 5-2. 과제 목표

아래 세 가지를 직접 수행합니다.

1. 하나의 모델 개선 목표를 정한다
2. 그것을 SFT용 데이터, DPO용 데이터, RL용 reward 문제로 각각 바꾼다
3. 각 방식의 장단점을 짧게 서술한다

## 5-3. 제공 자료

강의자가 아래 5개 프롬프트를 제공합니다.

* 질문 1: 고객 문의에 정중하게 답변하기
* 질문 2: 짧고 정확한 기술 요약 작성하기
* 질문 3: 간단한 산수 문제 풀기
* 질문 4: JSON 형식으로만 답하기
* 질문 5: 안전한 거절 응답 작성하기

## 5-4. 수강생 수행 항목

### Step 1. 목표 고르기

5개 중 2개를 선택합니다.

### Step 2. SFT 데이터 만들기

각 선택 문제에 대해 다음 형식으로 1건씩 작성합니다.

```yaml
instruction: "<사용자 질문>"
response: "<이상적인 답변>"
```

### Step 3. DPO 데이터 만들기

같은 문제에 대해 다음 형식으로 작성합니다.

```yaml
prompt: "<사용자 질문>"
chosen: "<더 선호되는 답변>"
rejected: "<덜 선호되는 답변>"
```

### Step 4. PPO/GRPO용 reward 규칙 만들기

같은 문제에 대해 reward rubric을 작성합니다.

```yaml
prompt: "<사용자 질문>"
reward_rules:
  - "<좋은 출력 조건 1>"
  - "<좋은 출력 조건 2>"
  - "<좋은 출력 조건 3>"
```

### Step 5. 방법 선택 메모

아래 질문에 5~7문장으로 답합니다.

* 이 문제는 SFT, DPO, PPO/GRPO 중 무엇이 가장 적합한가?
* 왜 그렇게 생각하는가?
* 데이터 확보 난이도는 어떤가?
* 평가 기준은 무엇인가?

---

## 5-5. 제출물 템플릿

### 제출물 1. 데이터 파일

파일명:
`module1_formats_<name>.json`

구성:

```json
{
  "task_1": {
    "sft": {},
    "dpo": {},
    "rl": {}
  },
  "task_2": {
    "sft": {},
    "dpo": {},
    "rl": {}
  }
}
```

### 제출물 2. 비교 메모

파일명:
`module1_method_selection_<name>.md`

형식:

```md
# Task 1
- Best method:
- Why:
- Data difficulty:
- Evaluation idea:

# Task 2
- Best method:
- Why:
- Data difficulty:
- Evaluation idea:
```

---

## 5-6. 평가 기준

### 우수

* SFT / DPO / RL 포맷 차이를 정확히 구분함
* chosen/rejected가 선호 차이를 분명히 드러냄
* reward 규칙이 측정 가능하게 작성됨
* 방법 선택 이유가 설득력 있음

### 보통

* 형식은 맞지만 chosen/rejected 차이가 약함
* reward 규칙이 추상적임
* 방법 선택 이유가 짧고 피상적임

### 미흡

* 세 방식의 차이를 거의 반영하지 못함
* DPO와 RL을 혼동함
* reward를 정량화할 수 없음

---

## 6. 강사용 해설 포인트

이 모듈에서 가장 중요한 것은 “정답”보다 **사고방식 전환**입니다.

예를 들어 수강생이
“친절한 답변을 만들고 싶다”라고 하면 강사는 반드시 다시 묻습니다.

* 친절함을 **정답 예시**로 줄 수 있는가 → SFT
* 친절한 답과 덜 친절한 답을 **비교쌍**으로 만들 수 있는가 → DPO
* 친절함을 **점수로 채점**할 수 있는가 → PPO/GRPO

이 세 질문을 반복하면, 수강생은 자연스럽게 각 방법의 데이터 철학을 구분하게 됩니다.

---

## 7. Module 1 종료 체크리스트

이 모듈이 끝날 때 수강생이 아래를 할 수 있으면 성공입니다.

* “DPO는 왜 reward model 없이도 선호 정렬을 할 수 있다고 하나요?”에 답할 수 있다. ([arXiv][2])
* “PPO는 왜 더 복잡한가요?”에 rollout, reward, surrogate update 관점에서 답할 수 있다. ([arXiv][3])
* “GRPO는 PPO와 어떤 관계인가요?”에 PPO 계열 변형이라는 수준으로 설명할 수 있다. ([arXiv][4])
* 같은 문제를 SFT, DPO, RL 문제로 각각 바꿔 쓸 수 있다.

---

## 8. 다음 Module과의 연결

Module 1은 아직 “개념과 데이터 사고” 단계입니다.
**Module 2**에서는 실제로 SmolLM급 모델을 로드하고 baseline inference를 수행하면서, 오늘 만든 문제들이 실제 튜닝 실험의 입력으로 어떻게 연결되는지 보여주면 자연스럽습니다. TRL quickstart도 compact model을 이용한 빠른 실험을 강조합니다. ([Hugging Face][1])

원하시면 이어서 바로 **Module 2: 실습 환경 구축 + baseline 측정**까지 같은 형식으로 이어서 작성하겠습니다.

[1]: https://huggingface.co/docs/trl/quickstart?utm_source=chatgpt.com "Quickstart"
[2]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[3]: https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com "Proximal Policy Optimization Algorithms"
[4]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical ..."
