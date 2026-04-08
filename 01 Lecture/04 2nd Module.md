아래는 **Module 2: 실습 환경 구축 + Baseline 측정**의 상세 설계안입니다. 이 모듈은 Module 1에서 정리한 개념을 실제 실습 환경으로 옮기는 단계이며, 이후 SFT·DPO·PPO·GRPO 비교 실습의 **공통 출발점(base checkpoint + baseline 결과)** 을 만드는 것이 목적입니다. 현재 Hugging Face의 TRL은 SFT, DPO, PPO, GRPO 계열 학습을 지원하며, SmolLM2는 135M, 360M, 1.7B 크기의 경량 모델 계열로 소개됩니다. 또한 TRL quickstart와 관련 문서는 이러한 후속 학습 흐름을 전제로 구성되어 있습니다. ([Hugging Face][1])

# Module 2

**실습 환경 구축 + Baseline 측정**

## 1. 모듈 개요

이 모듈의 목표는 세 가지입니다.

첫째, **실습 가능한 최소 환경**을 구성합니다.
둘째, **작은 모델을 실제로 로드하고 추론**해 봅니다.
셋째, 이후 SFT·DPO·PPO·GRPO 실험과 비교할 수 있도록 **baseline 결과를 고정된 평가셋으로 측정**합니다.

이 모듈이 중요한 이유는, 뒤에서 어떤 튜닝을 하더라도 “원래 모델이 어느 정도였는지”가 없으면 개선 여부를 제대로 판단할 수 없기 때문입니다. 또한 TRL은 여러 trainer를 제공하지만, 실제 실습에서는 먼저 모델 로드·토크나이저·생성 설정·평가셋 정리가 되어 있어야 이후 단계가 매끄럽게 진행됩니다. ([Hugging Face][1])

### 권장 시간

총 3시간 기준으로 설계합니다.

* 강의 50분
* 설치/환경 점검 30분
* 데모 20분
* 실습 60분
* 정리 20분

### 선수 지식

* Python 기초
* Jupyter/Colab 기초
* pip 설치 경험
* GPU와 CPU의 차이를 아주 기본 수준으로 이해하면 충분

### 모듈 산출물

모듈 종료 시 수강생은 아래 4가지를 제출할 수 있어야 합니다.

1. 실행 가능한 실습 환경
2. baseline inference 노트북 또는 스크립트
3. 공통 평가셋에 대한 baseline 출력 결과
4. 실험 로그 요약표

---

## 2. 학습 목표

수강생은 모듈 종료 시 다음을 수행할 수 있어야 합니다.

* Hugging Face 기반 실습 환경을 구성할 수 있다
* 소형 causal LM을 로드하고 tokenizer와 함께 추론할 수 있다
* 채팅형 입력을 일정한 템플릿으로 넣고 출력을 비교할 수 있다
* 고정된 프롬프트 세트로 baseline을 측정할 수 있다
* 이후 SFT·DPO·PPO·GRPO 실습에 사용할 공통 입력 데이터를 준비할 수 있다

Transformers 문서는 `AutoModelForCausalLM`과 tokenizer를 사용한 일반적인 생성 예시를 제공하고, SmolLM2 모델 카드는 이 계열이 경량 실습용으로 적합한 소형 모델군임을 설명합니다. ([Hugging Face][2])

---

## 3. 교안 초안

## 3-1. 슬라이드 구성안

### Slide 1. 모듈 제목

**“실습 환경 구축과 Baseline 측정”**

강사 멘트:

> 오늘은 튜닝을 아직 하지 않습니다. 대신 이후 모든 튜닝 결과와 비교할 기준점을 만듭니다.

---

### Slide 2. 왜 Baseline이 필요한가

내용:

* 튜닝 전 성능이 기준선
* 스타일, 정답률, 형식 준수율은 모두 “전/후 비교”가 중요
* baseline이 없으면 개선인지 착시인지 구분하기 어렵다

설명 포인트:

* “좋아진 것 같다”가 아니라 “무엇이 얼마나 좋아졌는가”로 바꾸기
* Module 5, 8의 비교 실험을 위해 반드시 필요

---

### Slide 3. 이번 실습의 전체 흐름

내용:
**환경 설치 → 모델 선택 → 모델 로드 → 생성 테스트 → 평가셋 준비 → baseline 저장**

강조:

* 오늘 만든 baseline 출력은 앞으로 계속 재사용
* SFT, DPO, PPO, GRPO의 공통 비교 기준이 됨

---

### Slide 4. 왜 SmolLM급 소형 모델을 쓰는가

내용:

* 작은 GPU 또는 Colab 환경에서도 반복 실험 가능
* 설치와 추론 속도가 상대적으로 가벼움
* 교육용으로 full pipeline 이해에 적합
* SmolLM2는 135M, 360M, 1.7B 크기로 제공됨

SmolLM2 모델 카드는 이 계열을 on-device까지 염두에 둔 compact family로 설명합니다. ([Hugging Face][3])

---

### Slide 5. 실습에 필요한 핵심 라이브러리

내용:

* `transformers`
* `datasets`
* `trl`
* `accelerate`
* `peft`
* 선택: `bitsandbytes`

TRL 문서는 SFT, DPO, PPO, GRPO 등을 포함하는 full-stack post-training 라이브러리로 설명하며, transformers와 통합되어 있습니다. ([Hugging Face][1])

---

### Slide 6. 오늘은 어떤 모델을 쓸까

권장안:

* 입문 실습: `HuggingFaceTB/SmolLM2-360M-Instruct` 급
* 여유 있는 GPU: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

강의 포인트:

* 너무 작은 모델은 품질이 낮을 수 있고
* 너무 큰 모델은 실습 반복이 어렵다
* 교육용으로는 “반복 가능한 크기”가 더 중요하다

SmolLM2 계열은 135M, 360M, 1.7B 세 가지 크기로 제공됩니다. ([Hugging Face][3])

---

### Slide 7. Base 모델과 Instruct 모델의 차이

내용:

* base model: 일반 언어 모델
* instruct model: instruction-following이 반영된 버전
* 교육 실습에서는 보통 instruct 모델로 시작하는 것이 안정적

SmolLM2 관련 카드에서는 instruct 계열이 rewriting, summarization, function calling 같은 작업 지원이 강화된다고 설명합니다. ([Hugging Face][4])

---

### Slide 8. 생성 파이프라인의 기본 구조

내용:

* tokenizer 로드
* model 로드
* prompt 구성
* `generate()`
* decode 및 결과 비교

Transformers 문서는 pipeline 또는 `AutoModelForCausalLM` 기반 텍스트 생성 예시를 제공합니다. ([Hugging Face][2])

---

### Slide 9. Baseline 측정용 공통 평가셋 설계

내용:
평가셋은 최소 4개 축으로 구성

* Persona/톤
* 요약/설명
* 수학/정답 검증
* 포맷 준수(JSON 등)

강의 포인트:

* Module 1의 “같은 문제를 다른 학습 방식으로 바꾸기”를 기억
* baseline도 그 축에 맞춰 측정해야 이후 비교가 가능

---

### Slide 10. 평가할 때 반드시 고정해야 하는 것

내용:

* 모델 이름
* temperature
* max_new_tokens
* system/user prompt 형식
* evaluation prompt set
* random seed 가능 시 고정

설명 포인트:

* 튜닝 전후 비교를 하려면 입력 조건을 최대한 고정
* prompt가 바뀌면 모델 변화가 아니라 실험 조건 변화일 수 있음

---

### Slide 11. 오늘의 실습 산출물

내용:

* 설치 성공 여부
* 모델 로드 성공
* 10개 공통 프롬프트에 대한 baseline 결과
* 관찰 메모

  * 강점
  * 약점
  * 다음 모듈에서 튜닝하고 싶은 항목

---

### Slide 12. 다음 모듈과의 연결

내용:

* Module 3에서는 오늘의 baseline 과제들을

  * SFT용
  * DPO용
  * PPO/GRPO용
    데이터셋으로 변환
* 즉 오늘의 결과는 이후 모든 모듈의 출발점

---

## 3-2. 강의 진행 시나리오

### Part A. 환경 설명 15분

강사는 실습이 돌아가는 최소 스택을 설명합니다.

핵심 메시지:

* 오늘은 “잘 학습시키는 법”보다 “안정적으로 실험을 시작하는 법”이 중요
* 도구를 많이 깔기보다, 먼저 하나의 모델을 확실히 로드하고 baseline을 얻는 것이 목표

### Part B. 모델 선택 설명 10분

세 가지 선택 기준을 설명합니다.

* 반복 실험 가능성
* instruction-following 품질
* 이후 튜닝 비용

강사는 기본 권장 모델을 하나 정해 주는 것이 좋습니다.
예를 들어 교육용 기본값은 360M급 instruct 모델, 옵션으로 1.7B급 instruct 모델을 제시합니다. ([Hugging Face][3])

### Part C. 데모 20분

강사가 직접 보여줄 내용

1. 라이브러리 설치
2. 모델 및 tokenizer 로드
3. 한두 개 prompt 생성
4. baseline 저장

### Part D. 수강생 실습 40분

수강생은 각자 환경에서 같은 흐름을 반복합니다.

### Part E. 결과 토론 15분

같은 모델인데도

* 생성 길이
* 말투
* 형식 준수
* 수학 정확도
  가 어떻게 달라 보이는지 चर्चा합니다.

---

## 4. 실습 예제 코드 주제

## 4-1. 코드 주제 A

**`setup_and_baseline.ipynb`**

목적:
환경 설치부터 모델 로드, 생성, baseline 저장까지 한 번에 수행하는 기본 노트북

핵심 셀 구성:

1. 패키지 설치
2. 모델명 설정
3. tokenizer / model 로드
4. 단일 프롬프트 생성
5. 공통 평가셋 로드
6. 배치 추론 또는 반복 추론
7. 결과 저장

예시 코드 주제:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

prompt = "사용자 질문에 친절한 한국어로 짧게 답하세요: 포스트 트레이닝이 무엇인가요?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

이 코드는 transformers의 일반적 causal LM 추론 패턴에 맞습니다. ([Hugging Face][2])

---

## 4-2. 코드 주제 B

**`baseline_eval_runner.py`**

목적:
고정된 prompt set으로 baseline 결과를 JSON 파일에 저장

입력:

* 모델명
* 평가 prompt 파일
* generation 설정

출력:

* `baseline_outputs.json`

추천 JSON 구조:

```json
{
  "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "generation_config": {
    "max_new_tokens": 128,
    "temperature": 0.7
  },
  "results": [
    {
      "task_id": "persona_01",
      "prompt": "고객 문의에 정중하게 답하세요...",
      "output": "안녕하세요..."
    }
  ]
}
```

---

## 4-3. 코드 주제 C

**`baseline_scorecard.py`**

목적:
정량화가 쉬운 일부 항목을 rule-based로 빠르게 측정

예:

* 출력 길이
* JSON 형식 파싱 성공 여부
* 산수 정답 일치 여부
* 지정 키워드 포함 여부

이 스크립트는 아직 완전한 평가가 아니라, Module 5에서 확장될 **baseline scorecard의 초안**입니다.

예시 코드 주제:

```python
import json

def is_json_response(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False

def exact_match(answer: str, target: str) -> bool:
    return answer.strip() == target.strip()

def response_length(text: str) -> int:
    return len(text)
```

---

## 5. Module 2 실습 과제 템플릿

## 5-1. 과제 제목

**실습 환경 구축 및 Baseline 결과 확보**

## 5-2. 과제 목표

아래 네 가지를 완료합니다.

1. 최소 실습 환경 설치
2. 지정 모델 로드
3. 공통 평가셋에 대한 baseline 생성
4. 결과 관찰 및 다음 튜닝 목표 정의

---

## 5-3. 제공 자료

강의자는 아래 자료를 제공합니다.

### 제공 파일 1. `requirements_module2.txt`

예시 패키지:

```txt
transformers
datasets
trl
accelerate
peft
```

TRL은 여러 post-training trainer를 제공하고, 이후 모듈에서 SFT, DPO, PPO, GRPO를 사용할 예정이므로 초기에 함께 설치해 두는 것이 자연스럽습니다. ([Hugging Face][1])

### 제공 파일 2. `baseline_prompts.json`

예시:

```json
[
  {
    "task_id": "persona_01",
    "category": "persona",
    "prompt": "고객 문의에 정중한 한국어로 3문장 이내로 답하세요: 배송이 늦어지고 있습니다."
  },
  {
    "task_id": "math_01",
    "category": "math",
    "prompt": "27 * 14 의 결과만 답하세요."
  },
  {
    "task_id": "format_01",
    "category": "format",
    "prompt": "name, role 키를 가진 JSON만 출력하세요. name은 Alice, role은 engineer."
  }
]
```

### 제공 파일 3. `baseline_observation_template.md`

관찰 기록용 문서

---

## 5-4. 수강생 수행 항목

### Step 1. 환경 설치

필수 라이브러리를 설치합니다.

예시:

```bash
pip install -U transformers datasets trl accelerate peft
```

### Step 2. 모델 선택

아래 중 하나를 선택합니다.

* 기본 트랙: SmolLM2 360M/1.7B instruct 계열
* 선택 트랙: 강사가 지정한 다른 소형 instruct 모델

교육용 권장 기본값은 SmolLM2 instruct 계열입니다. ([Hugging Face][3])

### Step 3. 모델 로드 및 단일 생성 확인

한 개 prompt에 대해 정상 출력이 나오는지 확인합니다.

### Step 4. 공통 평가셋 실행

`baseline_prompts.json`의 모든 항목에 대해 출력 생성

### Step 5. 결과 저장

아래 형식으로 저장합니다.

```json
{
  "model_name": "...",
  "results": [
    {
      "task_id": "...",
      "output": "..."
    }
  ]
}
```

### Step 6. 관찰 메모 작성

아래 질문에 답합니다.

* 가장 잘 되는 항목은 무엇인가
* 가장 약한 항목은 무엇인가
* 스타일 튜닝이 먼저 필요한가, 능력 튜닝이 먼저 필요한가
* Module 3에서 어떤 데이터셋을 만들고 싶은가

---

## 5-5. 제출물 템플릿

### 제출물 1. 실행 결과 파일

파일명:
`module2_baseline_outputs_<name>.json`

형식:

```json
{
  "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "generation_config": {
    "max_new_tokens": 128,
    "temperature": 0.7
  },
  "results": [
    {
      "task_id": "persona_01",
      "category": "persona",
      "prompt": "....",
      "output": "...."
    }
  ]
}
```

### 제출물 2. 관찰 메모

파일명:
`module2_baseline_observations_<name>.md`

형식:

```md
# Baseline Observation

## Strongest category
-

## Weakest category
-

## Notable behavior
-

## What should be improved first?
-

## Candidate method for improvement
- SFT / DPO / PPO / GRPO
- reason:
```

### 제출물 3. 실행 환경 정보

파일명:
`module2_env_<name>.txt`

예시:

```txt
python version:
torch version:
transformers version:
trl version:
device:
model_name:
```

---

## 5-6. 평가 기준

### 우수

* 환경 설치와 모델 로드가 완전히 성공
* 모든 baseline prompt에 대해 결과 저장 완료
* generation 설정과 모델명을 명확히 기록
* 관찰 메모가 구체적이며 다음 모듈과 연결됨

### 보통

* 실행은 되었으나 일부 prompt 누락
* 환경 정보 기록이 불완전
* 관찰 메모가 추상적임

### 미흡

* baseline 결과를 재현할 수 없게 저장
* 모델명 또는 generation 설정 누락
* 출력이 있어도 비교 기준이 없음

---

## 6. 강사용 해설 포인트

이 모듈에서 강사가 계속 강조해야 할 점은 두 가지입니다.

첫째, **오늘의 목표는 성능 최대화가 아니라 기준선 확보**입니다.
수강생이 “더 좋은 모델로 바꿔도 되나요?”라고 물으면, 강사는 가능하면 같은 기본 모델을 유지하게 해야 합니다. 그래야 뒤 모듈의 SFT, DPO, PPO, GRPO 비교가 의미 있어집니다.

둘째, **baseline은 평가셋과 생성설정까지 포함해야 한다**는 점입니다.
모델 이름만 저장하고 temperature, max tokens, prompt 형식을 기록하지 않으면 비교 실험이 흐트러집니다.

강사가 던지면 좋은 질문:

* 이 모델은 정중함은 괜찮은데 JSON 형식은 왜 약할까?
* 수학 문제는 틀리는데 설명은 그럴듯한 이유가 무엇일까?
* 이 약점을 SFT로 먼저 고칠지, DPO로 정렬할지, RL로 보상 최적화할지 어떻게 판단할까?

---

## 7. Module 2 종료 체크리스트

이 모듈이 끝나면 수강생은 아래를 할 수 있어야 합니다.

* Hugging Face 모델을 직접 로드하고 텍스트 생성 가능
* tokenizer, model, generation config의 역할을 구분 가능
* 공통 평가 prompt set을 이용해 baseline 결과를 확보 가능
* 이후 모듈에서 사용할 비교 기준선을 저장 가능
* “어떤 약점을 어떤 튜닝 방식으로 개선할지”에 대한 초기 가설을 세울 수 있음

TRL과 transformers 문서는 이런 후속 학습을 위한 공통 기반 환경 구성을 전제로 하고 있으며, TRL은 이후 SFT, DPO, PPO, GRPO trainer를 같은 생태계 안에서 이어서 사용할 수 있게 설계되어 있습니다. ([Hugging Face][1])

---

## 8. 다음 Module과의 연결

**Module 3**에서는 오늘 확보한 baseline prompt와 출력 결과를 바탕으로, 같은 과제를

* SFT용 `instruction-response`
* DPO용 `prompt-chosen-rejected`
* PPO/GRPO용 `prompt + reward rubric`
  형태로 변환하는 **데이터 큐레이션 실습**으로 넘어가면 가장 자연스럽습니다. TRL의 dataset formats 문서도 task별 데이터 타입이 다르며, prompt-only, preference, language modeling 등으로 구분된다고 설명합니다. ([Hugging Face][5])

원하시면 이어서 바로 **Module 3: 데이터 큐레이션과 학습 데이터 설계**를 같은 형식으로 작성하겠습니다.

[1]: https://huggingface.co/docs/trl/index?utm_source=chatgpt.com "TRL - Transformers Reinforcement Learning"
[2]: https://huggingface.co/docs/transformers/model_doc/nemotron_h?utm_source=chatgpt.com "NemotronH"
[3]: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct?utm_source=chatgpt.com "HuggingFaceTB/SmolLM2-1.7B-Instruct"
[4]: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B?utm_source=chatgpt.com "HuggingFaceTB/SmolLM2-1.7B"
[5]: https://huggingface.co/docs/trl/en/dataset_formats?utm_source=chatgpt.com "Dataset formats and types"
