Module 3를 실제 강의용으로 바로 쓸 수 있게, 교안 구조와 함께 SFT/DPO/PPO/GRPO 데이터 변환 실습 중심으로 정리
현재 TRL 기준 데이터 포맷과 각 trainer가 기대하는 입력 형태를 다시 확인한 뒤, baseline prompt를 어떻게 네 갈래로 바꾸는지까지 연결해서 작성


이 모듈의 핵심은 “학습 알고리즘”이 아니라 데이터 관점의 사고 전환

TRL 문서 기준으로 SFT는 language modeling 또는 prompt-completion 형식을 받고, DPO는 preference 형식인 prompt / chosen / rejected를 기대합니다. TRL의 dataset guide는 데이터 타입을 language modeling, prompt-only, prompt-completion, preference 등으로 구분하며, GRPO는 최소 "prompt" 컬럼이 있는 데이터셋과 reward function을 사용합니다. PPO 문서는 RL 루프를 rollout → evaluation → optimization으로 설명하고, 먼저 SFT 모델과 reward model 또는 reward signal을 준비하는 흐름을 권장합니다.


좋습니다.
아래는 **Module 3: 지금 사용한 baseline prompt를 SFT/DPO/PPO/GRPO 데이터로 변환**에 대한 **강의 교안 + 실습 예제** 완성 초안입니다.

이 모듈의 핵심은 “학습 알고리즘”이 아니라 **데이터 관점의 사고 전환**입니다.
TRL 문서 기준으로 SFT는 `language modeling` 또는 `prompt-completion` 형식을 받고, DPO는 `preference` 형식인 `prompt / chosen / rejected`를 기대합니다. TRL의 dataset guide는 데이터 타입을 `language modeling`, `prompt-only`, `prompt-completion`, `preference` 등으로 구분하며, GRPO는 최소 `"prompt"` 컬럼이 있는 데이터셋과 reward function을 사용합니다. PPO 문서는 RL 루프를 `rollout → evaluation → optimization`으로 설명하고, 먼저 SFT 모델과 reward model 또는 reward signal을 준비하는 흐름을 권장합니다. ([Hugging Face][1])

---

# Module 3

**Baseline Prompt를 SFT / DPO / PPO / GRPO 데이터로 변환하기**

## 1. 모듈 개요

Module 2에서 우리는 baseline prompt와 baseline output을 확보했습니다.
Module 3에서는 그 결과를 바탕으로 **같은 문제를 서로 다른 post-training 데이터셋 형식으로 재구성**합니다. 이 과정이 중요한 이유는, 같은 목표라도 SFT는 “정답 예시”, DPO는 “선호 비교쌍”, PPO/GRPO는 “보상 계산 가능한 문제”로 바꿔야 하기 때문입니다. TRL의 dataset guide도 trainer마다 기대하는 데이터 타입이 다르다고 명시합니다. ([Hugging Face][1])

이 모듈에서 수강생은 아래를 직접 하게 됩니다.

* Module 2의 `baseline_prompts.json` 또는 `module2_baseline_outputs.json` 읽기
* 각 prompt를 **SFT용 `prompt-completion` 데이터**로 바꾸기
* 같은 prompt를 **DPO용 `prompt-chosen-rejected` 데이터**로 바꾸기
* 같은 prompt를 **PPO용 prompt + reward metadata 데이터**로 바꾸기
* 같은 prompt를 **GRPO용 prompt + additional columns 데이터**로 바꾸기

여기서 특히 중요한 점은, **PPO와 GRPO는 “정답 텍스트”를 저장하는 것이 아니라 나중에 생성 결과를 점수화할 수 있도록 보상 기준을 데이터에 심어두는 방식**으로 설계하는 것이 자연스럽다는 점입니다. 이는 PPO 문서의 RL loop 설명과, GRPO 문서의 custom reward function이 `prompts`, `completions`, 그리고 추가 컬럼을 받을 수 있다는 설명에 부합합니다. ([Hugging Face][2])

### 권장 시간

총 3시간 기준입니다.

* 강의 60분
* 데모 20분
* 실습 70분
* 리뷰 30분

### 모듈 산출물

모듈 종료 시 수강생은 아래 4개 파일을 만들 수 있어야 합니다.

* `module3_sft_dataset.jsonl`
* `module3_dpo_dataset.jsonl`
* `module3_ppo_dataset.jsonl`
* `module3_grpo_dataset.jsonl`

---

## 2. 학습 목표

이 모듈이 끝나면 수강생은 다음을 설명할 수 있어야 합니다.

* 왜 같은 baseline prompt가 학습 방식마다 다른 데이터 구조를 가져야 하는가
* SFT용 정답 예시와 DPO용 선호쌍의 차이
* PPO용 데이터 설계에서 왜 reward rubric 또는 ground truth가 중요한가
* GRPO용 데이터 설계에서 왜 `"prompt"`와 추가 컬럼이 핵심인가
* baseline output을 단순 저장물이 아니라 **후속 데이터셋 생성 재료**로 활용하는 방법

이 학습 목표는 TRL의 SFT, DPO, dataset formats, GRPO docs에서 설명하는 기대 입력 형식을 교육용 실습 흐름으로 재구성한 것입니다. ([Hugging Face][1])

---

## 3. 강의 교안 초안

## 3-1. 슬라이드 구성안

### Slide 1. 제목

**“같은 문제를 다른 학습 데이터로 바꾸기”**

강사 멘트:
오늘의 목표는 prompt를 더 많이 만드는 것이 아니라, **같은 prompt를 4개의 학습 철학으로 바꾸는 것**입니다.

---

### Slide 2. Module 2에서 가져오는 것

내용:

* baseline prompt
* baseline model output
* category 정보
* baseline 관찰 메모

설명:

* prompt 자체도 중요하지만
* baseline output은 `rejected` 후보나 error pattern 추출에 매우 유용함

---

### Slide 3. TRL 기준 데이터 타입 지도

내용:

* SFT → `prompt-completion` 또는 `language modeling`
* DPO → `preference`
* PPO/GRPO → prompt에서 rollout 후 reward 계산
* GRPO → `"prompt"` 필수, 추가 컬럼은 reward function에 전달 가능

TRL의 dataset guide는 `prompt-completion`과 `preference`를 구분하고, GRPO docs는 train dataset에 `"prompt"` 컬럼이 있어야 하며 추가 컬럼은 custom reward function으로 전달될 수 있다고 설명합니다. ([Hugging Face][1])

---

### Slide 4. SFT용으로 바꾸는 법

내용:

* 한 문제당 “이상적인 답변” 1개 이상 작성
* `prompt`와 `completion`으로 저장
* 또는 conversational 형식으로 저장 가능
* 목표: 스타일, 형식, 기본 능력 주입

SFTTrainer는 `language modeling`과 `prompt-completion` 형식을 지원하고, standard와 conversational format 모두 처리할 수 있습니다. ([Hugging Face][3])

---

### Slide 5. DPO용으로 바꾸는 법

내용:

* 같은 prompt에 대해 `chosen`과 `rejected`를 만든다
* explicit `prompt`를 분리하는 방식이 권장됨
* 목표: 상대 선호를 직접 학습

DPOTrainer는 `preference` 데이터셋을 기대하며, docs는 explicit prompt 형식을 권장합니다. ([Hugging Face][4])

---

### Slide 6. PPO용으로 바꾸는 법

내용:

* prompt 자체가 출발점
* 정답 텍스트보다 reward 계산 가능성이 중요
* 예: ground truth, format rule, safety rule, target length

PPO docs는 PPO를 `rollout → evaluation → optimization`으로 설명하고, reward는 handcrafted rule, metric, reward model 등에서 올 수 있다고 설명합니다. 또한 PPO 전에 SFT 모델을 먼저 만드는 흐름을 권장합니다. ([Hugging Face][2])

---

### Slide 7. GRPO용으로 바꾸는 법

내용:

* `"prompt"` 필수
* 정답/형식 기준 같은 보조 컬럼 추가 가능
* custom reward function이 `prompts`, `completions`, 추가 컬럼을 입력받아 점수 계산
* reasoning, correctness, format 보상 설계에 잘 맞음

GRPO docs는 custom reward function이 `prompts`, `completions`, `completions_ids`, `trainer_state`, 그리고 데이터셋의 추가 컬럼을 받을 수 있다고 설명합니다. train dataset은 `"prompt"` 컬럼을 포함해야 합니다. ([Hugging Face][5])

---

### Slide 8. 예시 1: JSON 출력 과제

기존 baseline prompt:
“name, role 키를 가진 JSON만 출력하세요…”

변환:

* SFT: 이상적인 JSON 답 1개
* DPO: parse되는 JSON vs 설명문이 섞인 응답
* PPO: `must_be_json=True`, `required_keys=["name","role"]`
* GRPO: `"prompt"` + `"required_keys"` + `"task_type":"format"`

---

### Slide 9. 예시 2: 수학 과제

기존 baseline prompt:
“27 * 14 의 결과만 답하세요.”

변환:

* SFT: `"378"`
* DPO: `"378"` vs `"답은 378입니다."`
* PPO: `ground_truth=378`, `answer_type="integer_only"`
* GRPO: `"prompt"` + `"ground_truth": "378"` + `"task":"math"`

---

### Slide 10. 왜 baseline output이 중요한가

내용:

* baseline이 틀린 답을 자주 만든다면 DPO의 `rejected` 후보가 된다
* baseline이 형식을 어긴다면 format penalty rule의 근거가 된다
* baseline이 장황하다면 brevity reward 설계에 도움이 된다

즉 Module 2 출력은 단순 관찰 자료가 아니라 **데이터 생성기**입니다.

---

### Slide 11. 이 모듈의 핵심 판단 질문

내용:

* “정답을 줄 수 있는가?” → SFT
* “둘 중 무엇이 더 낫다고 말할 수 있는가?” → DPO
* “출력을 점수화할 수 있는가?” → PPO / GRPO

---

### Slide 12. 다음 모듈 연결

내용:

* Module 4에서는 오늘 만든 SFT 데이터로 실제 SFT 진행
* Module 5 이후 DPO / PPO / GRPO 실험으로 확장

---

## 3-2. 강의 진행 시나리오

### Part A. 개념 정리 15분

강사는 먼저 TRL의 데이터 타입 지도를 보여주고, “trainer가 바뀌면 데이터가 바뀐다”는 점을 강조합니다. SFT는 `prompt-completion`, DPO는 `preference`, GRPO는 `"prompt"`와 reward function 중심이라는 점을 분명히 짚습니다. ([Hugging Face][1])

### Part B. baseline 재해석 15분

Module 2의 출력 파일을 열고, baseline이 잘한 점과 못한 점을 봅니다.
그다음 “이 오답을 DPO의 rejected로 쓸 수 있는가?”, “이 실패를 reward rule로 바꿀 수 있는가?”를 묻습니다.

### Part C. 형식 변환 데모 20분

강사가 한 개 task를 실시간으로 4개 데이터 형식으로 변환합니다.

예:

* prompt: JSON 출력 과제
* SFT row 1개
* DPO row 1개
* PPO row 1개
* GRPO row 1개

### Part D. 수강생 실습 50분

수강생은 baseline task 4~8개를 선택해 직접 변환 파일을 만듭니다.

### Part E. 리뷰 20분

같은 prompt라도 왜 “정답 텍스트”와 “보상 기준”이 다르게 쓰이는지 토론합니다.

---

## 4. 실습 예제 코드 주제

이번 모듈의 실습 코드는 **“Module 2 결과를 읽어서 4개 데이터셋으로 변환하는 데이터 엔지니어링 실습”**으로 설계하는 것이 가장 좋습니다.

## 4-1. 코드 주제 A

**`load_module2_baseline.py`**

목적:

* `module2_baseline_outputs.json` 읽기
* task별 prompt/category/output 추출
* 이후 변환 함수의 입력으로 사용

```python
import json

with open("/content/module2_outputs/module2_baseline_outputs.json", "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

rows = baseline_data["results"]
print("num rows:", len(rows))
print(rows[0])
```

이 코드는 Module 2 산출물을 Module 3의 입력으로 연결하는 출발점입니다.

---

## 4-2. 코드 주제 B

**`build_sft_dataset.py`**

목적:
baseline prompt를 SFT용 `prompt-completion` 형식으로 바꾸기

강의 포인트:

* baseline output을 그대로 쓰지 말고
* **이상적인 답변**을 사람이 보정해서 completion으로 넣는다

예시 코드:

```python
def build_sft_example(row):
    prompt = row["prompt"]
    category = row["category"]

    if category == "math":
        if row["task_id"] == "math_01":
            completion = "378"
        elif row["task_id"] == "math_02":
            completion = "503"
        else:
            completion = row["output"]

    elif category == "format":
        if row["task_id"] == "format_01":
            completion = '{"name": "Alice", "role": "engineer"}'
        elif row["task_id"] == "format_02":
            completion = '{"title": "DPO Basics", "difficulty": "beginner"}'
        else:
            completion = row["output"]

    elif category == "persona":
        completion = "안녕하세요. 배송이 지연되어 불편을 드려 죄송합니다. 현재 상황을 확인 중이며 빠르게 안내드리겠습니다."

    elif category == "summary":
        completion = "포스트 트레이닝은 사전학습된 모델을 특정 목적에 맞게 추가 조정하는 과정입니다. 이를 통해 스타일, 선호, 과제 수행 능력을 개선할 수 있습니다."

    elif category == "safety":
        completion = "그 요청은 도와드릴 수 없습니다. 대신 합법적이고 안전한 보안 학습 방법은 안내드릴 수 있습니다."

    else:
        completion = row["output"]

    return {
        "prompt": prompt,
        "completion": completion
    }
```

SFTTrainer는 `prompt-completion` 형식을 직접 지원합니다. ([Hugging Face][3])

---

## 4-3. 코드 주제 C

**`build_dpo_dataset.py`**

목적:
baseline prompt를 DPO용 `prompt-chosen-rejected` 형식으로 바꾸기

강의 포인트:

* `chosen`은 더 선호되는 답
* `rejected`는 단순 오답뿐 아니라

  * 장황한 답
  * 형식 위반 답
  * 덜 공손한 답
  * 설명이 섞여 있는 답
    도 가능

예시 코드:

```python
def build_dpo_example(row):
    prompt = row["prompt"]
    baseline_output = row["output"]
    category = row["category"]

    if category == "math":
        chosen = "378" if row["task_id"] == "math_01" else "503"
        rejected = baseline_output if baseline_output.strip() != chosen else f"답은 {chosen}입니다."

    elif category == "format":
        if row["task_id"] == "format_01":
            chosen = '{"name": "Alice", "role": "engineer"}'
            rejected = 'Alice는 engineer입니다.'
        else:
            chosen = '{"title": "DPO Basics", "difficulty": "beginner"}'
            rejected = '제목은 DPO Basics이고 난이도는 beginner입니다.'

    elif category == "persona":
        chosen = "안녕하세요. 배송 지연으로 불편을 드려 죄송합니다. 현재 확인 중이며 빠르게 안내드리겠습니다."
        rejected = "배송이 늦어지고 있습니다. 기다려 주세요."

    elif category == "summary":
        chosen = "포스트 트레이닝은 사전학습 모델을 목적에 맞게 추가 조정하는 과정입니다. 이를 통해 스타일과 성능을 개선할 수 있습니다."
        rejected = "포스트 트레이닝은 여러 단계를 포함하며 다양한 알고리즘과 환경 구성, 평가 절차가 필요합니다."

    elif category == "safety":
        chosen = "그 요청은 도와드릴 수 없습니다. 대신 안전한 보안 학습 방법은 안내할 수 있습니다."
        rejected = "해킹 방법을 단계별로 알려드릴게요."

    else:
        chosen = baseline_output
        rejected = baseline_output + "."

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }
```

DPOTrainer는 `prompt`, `chosen`, `rejected`를 갖는 explicit preference 형식을 권장합니다. ([Hugging Face][4])

---

## 4-4. 코드 주제 D

**`build_ppo_dataset.py`**

목적:
PPO용 source dataset 만들기

여기서 중요한 점은, PPO 실습에서는 보통 **정답 텍스트 자체를 바로 학습시키지 않고**, prompt에서 rollout을 한 뒤 reward를 계산합니다. PPO docs도 reward가 handcrafted rule, metric, reward model 등에서 올 수 있다고 설명합니다. 따라서 교육용 데이터셋은 `prompt`와 reward 계산용 metadata를 함께 저장하는 구조가 적합합니다. ([Hugging Face][2])

예시 코드:

```python
def build_ppo_example(row):
    prompt = row["prompt"]
    category = row["category"]

    example = {
        "prompt": prompt,
        "task_type": category
    }

    if category == "math":
        example["ground_truth"] = "378" if row["task_id"] == "math_01" else "503"
        example["reward_mode"] = "exact_match_integer"

    elif category == "format":
        if row["task_id"] == "format_01":
            example["required_keys"] = ["name", "role"]
        else:
            example["required_keys"] = ["title", "difficulty"]
        example["reward_mode"] = "json_schema"

    elif category == "persona":
        example["must_include"] = ["안녕하세요"]
        example["max_chars"] = 180
        example["reward_mode"] = "polite_and_brief"

    elif category == "safety":
        example["must_refuse"] = True
        example["reward_mode"] = "safe_refusal"

    return example
```

이건 “TRL PPOTrainer가 반드시 이 JSON 구조를 그대로 요구한다”는 의미가 아니라, **PPO 실습에서 rollout 후 reward를 계산하기 위한 교육용 source dataset 설계**입니다. PPO 문서의 RL loop 설명에 맞춰 만든 과정용 포맷이라고 이해하면 됩니다. ([Hugging Face][2])

---

## 4-5. 코드 주제 E

**`build_grpo_dataset.py`**

목적:
GRPO용 source dataset 만들기

GRPO docs에 따르면 train dataset은 `"prompt"` 컬럼을 포함해야 하고, custom reward function은 추가 컬럼도 받을 수 있습니다. 따라서 실습용 데이터셋은 prompt-only를 기본으로 하되, 보상 계산에 필요한 `ground_truth`, `task_type`, `required_keys` 같은 컬럼을 함께 두는 구조가 좋습니다. ([Hugging Face][5])

예시 코드:

```python
def build_grpo_example(row):
    prompt = row["prompt"]
    category = row["category"]

    example = {
        "prompt": prompt,
        "task_type": category
    }

    if category == "math":
        example["ground_truth"] = "378" if row["task_id"] == "math_01" else "503"

    elif category == "format":
        example["required_keys"] = (
            ["name", "role"] if row["task_id"] == "format_01"
            else ["title", "difficulty"]
        )

    elif category == "persona":
        example["must_include"] = ["안녕하세요"]
        example["max_chars"] = 180

    elif category == "safety":
        example["must_refuse"] = True

    return example
```

GRPO의 custom reward function은 `prompts`, `completions`, 추가 컬럼을 입력받고 float reward list를 반환해야 합니다. 여러 reward function을 함께 쓰는 것도 가능합니다. ([Hugging Face][5])

---

## 4-6. 코드 주제 F

**`export_all_datasets.py`**

목적:
4개 형식으로 한 번에 저장

```python
import json

sft_dataset = [build_sft_example(r) for r in rows]
dpo_dataset = [build_dpo_example(r) for r in rows]
ppo_dataset = [build_ppo_example(r) for r in rows]
grpo_dataset = [build_grpo_example(r) for r in rows]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save_jsonl("/content/module3_sft_dataset.jsonl", sft_dataset)
save_jsonl("/content/module3_dpo_dataset.jsonl", dpo_dataset)
save_jsonl("/content/module3_ppo_dataset.jsonl", ppo_dataset)
save_jsonl("/content/module3_grpo_dataset.jsonl", grpo_dataset)

print("All datasets saved.")
```

---

## 5. Module 3 실습 과제 템플릿

### 과제 제목

**Module 2 baseline 결과를 4개 post-training 데이터셋으로 변환하기**

### 과제 목표

수강생은 최소 6개 baseline task를 골라 아래 작업을 수행합니다.

* SFT dataset 생성
* DPO dataset 생성
* PPO source dataset 생성
* GRPO source dataset 생성
* 각 데이터 형식의 설계 의도를 설명

### Step 1. baseline 결과 읽기

* `module2_baseline_outputs.json`을 불러옵니다.

### Step 2. task 선택

아래 category를 최소 1개씩 포함하도록 6개 이상 선택합니다.

* persona
* math
* format
* safety 또는 summary

### Step 3. SFT 데이터 만들기

각 task에 대해 ideal completion을 직접 씁니다.

제출 형식 예:

```json
{"prompt":"27 * 14 의 결과만 답하세요.","completion":"378"}
```

### Step 4. DPO 데이터 만들기

각 task에 대해 `chosen`과 `rejected`를 만듭니다.

제출 형식 예:

```json
{"prompt":"27 * 14 의 결과만 답하세요.","chosen":"378","rejected":"답은 378입니다."}
```

### Step 5. PPO 데이터 만들기

각 task에 대해 reward 계산용 metadata를 만듭니다.

제출 형식 예:

```json
{"prompt":"27 * 14 의 결과만 답하세요.","task_type":"math","ground_truth":"378","reward_mode":"exact_match_integer"}
```

### Step 6. GRPO 데이터 만들기

각 task에 대해 GRPO reward function이 사용할 보조 컬럼을 만듭니다.

제출 형식 예:

```json
{"prompt":"name, role 키를 가진 JSON만 출력하세요.","task_type":"format","required_keys":["name","role"]}
```

### Step 7. 방법 선택 메모 작성

아래 질문에 대해 task별로 3~5문장씩 씁니다.

* 이 task는 SFT만으로 충분한가?
* DPO가 더 잘 맞는 이유가 있는가?
* PPO/GRPO reward 설계가 가능한가?
* 실제 실험에선 어떤 방법을 먼저 쓰겠는가?

---

## 6. 제출물 템플릿

### 제출물 1. SFT 데이터셋

파일명:
`module3_sft_dataset_<name>.jsonl`

### 제출물 2. DPO 데이터셋

파일명:
`module3_dpo_dataset_<name>.jsonl`

### 제출물 3. PPO 데이터셋

파일명:
`module3_ppo_dataset_<name>.jsonl`

### 제출물 4. GRPO 데이터셋

파일명:
`module3_grpo_dataset_<name>.jsonl`

### 제출물 5. 설명 메모

파일명:
`module3_dataset_design_notes_<name>.md`

형식:

```md
# Task 1
- Why SFT format:
- Why DPO format:
- PPO reward idea:
- GRPO reward idea:
- Best method first:

# Task 2
...
```

---

## 7. 평가 기준

### 우수

* 4개 데이터셋이 명확히 다른 목적을 반영함
* DPO의 chosen/rejected 차이가 분명함
* PPO/GRPO 데이터에 reward 계산용 정보가 구체적으로 들어 있음
* 설명 메모가 실제 학습 전략으로 연결됨

### 보통

* 형식은 맞지만 SFT와 DPO 차이가 약함
* PPO와 GRPO가 사실상 같은 설명에 머묾
* reward metadata가 추상적임

### 미흡

* 네 형식의 차이를 거의 반영하지 못함
* DPO에 정답/오답만 넣고 선호 기준이 없음
* PPO/GRPO에서 점수화 기준이 없음

---

## 8. 강사용 해설 포인트

이 모듈에서 강사가 가장 강하게 잡아야 할 메시지는 하나입니다.

**“모델 학습법을 바꾸려면 먼저 데이터셋의 질문 방식을 바꿔야 한다.”**

예를 들어
“JSON만 출력하게 만들고 싶다”는 요구가 들어왔을 때:

* SFT 관점: 올바른 JSON 답변 예시를 충분히 준다
* DPO 관점: parse되는 JSON이 chosen, 설명이 섞인 응답이 rejected
* PPO 관점: parse 성공 여부를 reward로 준다
* GRPO 관점: `prompt`와 `required_keys`를 두고 reward function으로 채점한다

이 네 문장을 수강생이 말할 수 있게 되면 Module 3은 성공입니다. 이 구분은 TRL의 dataset type 구분과 DPO/GRPO의 기대 입력 설명, PPO의 RL loop 설명과 정확히 맞물립니다. ([Hugging Face][1])

---

## 9. Module 3 종료 체크리스트

이 모듈이 끝나면 수강생은 다음을 할 수 있어야 합니다.

* baseline prompt 하나를 보고 SFT row를 만들 수 있다
* 같은 prompt를 DPO `chosen/rejected`로 바꿀 수 있다
* 같은 prompt를 PPO용 reward metadata 문제로 바꿀 수 있다
* 같은 prompt를 GRPO용 `prompt + extra columns`로 바꿀 수 있다
* 다음 모듈에서 바로 사용할 dataset file을 저장할 수 있다

---

## 10. 다음 Module과의 연결

다음 **Module 4**에서는 오늘 만든 `module3_sft_dataset.jsonl`을 사용해 실제로 **SFTTrainer 기반 SFT 실습**으로 들어가면 가장 자연스럽습니다. TRL quickstart도 가장 먼저 SFTTrainer 예시를 제시하고, SFTTrainer는 `prompt-completion` 형식을 직접 지원합니다. ([Hugging Face][6])

원하시면 이어서 바로 **Module 3용 Colab notebook 교안형 실습본**까지 만들어 드리겠습니다.

[1]: https://huggingface.co/docs/trl/en/dataset_formats "Dataset formats and types · Hugging Face"
[2]: https://huggingface.co/docs/trl/v0.8.0/ppo_trainer "PPO Trainer · Hugging Face"
[3]: https://huggingface.co/docs/trl/sft_trainer "SFT Trainer · Hugging Face"
[4]: https://huggingface.co/docs/trl/dpo_trainer "DPO Trainer · Hugging Face"
[5]: https://huggingface.co/docs/trl/v0.22.1/en/grpo_trainer "GRPO Trainer · Hugging Face"
[6]: https://huggingface.co/docs/trl/quickstart "Quickstart · Hugging Face"


