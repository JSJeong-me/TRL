Module 4를 실제로 바로 실행 가능한 SFT 실습 중심으로 잡겠습니다. TRL의 현재 SFTTrainer 기준 입력 형식과 prompt-completion 처리 방식을 공식 문서 기준으로 확인한 뒤, module3_sft_dataset.jsonl을 바로 쓰는 Colab형 예제까지 정리



좋습니다.
**Module 4**는 지금까지 만든 흐름에서 가장 자연스럽게 **`module3_sft_dataset.jsonl` → `SFTTrainer` 기반 실제 SFT 실행**으로 이어지는 단계입니다. TRL 공식 quickstart도 가장 먼저 `SFTTrainer` 예시를 제시하고, 공식 SFT 문서는 `prompt-completion` 형식을 직접 지원한다고 설명합니다. 또한 `prompt-completion` 데이터셋을 쓰면 기본적으로 **completion 부분에만 loss**를 계산하고, 전체 시퀀스에 loss를 걸고 싶으면 `completion_only_loss=False`로 바꿀 수 있습니다. ([Hugging Face][1])

# Module 4

**`module3_sft_dataset.jsonl`을 이용한 SFTTrainer 기반 SFT 실습**

## 1. 이번 모듈의 목표

이번 모듈의 목표는 세 가지입니다.

* `module3_sft_dataset.jsonl`을 로드한다.
* `SFTTrainer`로 실제 SFT를 수행한다.
* 튜닝 전후 출력을 비교해, SFT가 말투·형식·기본 과제 수행을 어떻게 바꾸는지 확인한다.

TRL 공식 문서에 따르면 `SFTTrainer`는 `language modeling`과 `prompt-completion` 데이터셋을 모두 지원하고, 표준 문자열 형식과 conversational 형식 모두 처리할 수 있습니다. 또 `prompt`와 `completion`이 따로 주어지면 trainer가 이를 이어 붙여 토크나이즈합니다. ([Hugging Face][2])

## 2. 이번 모듈에서 사용할 데이터 형식

이번 실습은 Module 3에서 만든 표준 `prompt-completion` JSONL을 그대로 사용합니다.
TRL의 dataset formats 문서는 표준 `prompt-completion` 형식을 아래처럼 정의합니다. ([Hugging Face][3])

```json
{"prompt": "27 * 14 의 결과만 답하세요.", "completion": "378"}
{"prompt": "name, role 키를 가진 JSON만 출력하세요. name은 Alice, role은 engineer.", "completion": "{\"name\": \"Alice\", \"role\": \"engineer\"}"}
```

이 형식은 Module 4에 딱 맞습니다. 이유는 SFT가 “정답 예시를 보여주는 방식”이기 때문입니다. DPO처럼 `chosen/rejected`가 필요하지 않고, PPO/GRPO처럼 reward metadata도 아직 필요하지 않습니다. `prompt-completion`만 준비되어 있으면 바로 SFT를 시작할 수 있습니다. ([Hugging Face][3])

## 3. 강의에서 먼저 설명할 핵심 포인트

이번 모듈에서 반드시 잡아줘야 할 개념은 네 가지입니다.

첫째, **왜 SFT를 먼저 하느냐**입니다.
TRL quickstart는 대표 trainer 예시 중 맨 앞에 SFT를 배치하고 있고, 실제 post-training 파이프라인에서도 SFT는 기본 instruction-following, 말투, 형식, 작업 습관을 먼저 안정화하는 출발점으로 자주 쓰입니다. ([Hugging Face][1])

둘째, **`prompt-completion` 데이터의 장점**입니다.
이 형식은 사람이 보기 쉽고 직접 만들기 쉬우며, SFTTrainer가 바로 받아들일 수 있습니다. 공식 docs도 `SFTTrainer`가 `prompt-completion`을 지원한다고 명시합니다. ([Hugging Face][2])

셋째, **loss가 어디에 걸리는가**입니다.
공식 SFT 문서에 따르면 `prompt-completion` 데이터셋에서는 기본적으로 completion 토큰에만 loss를 계산합니다. 즉, “질문을 외우게 하는 것”보다 “답변을 어떻게 생성해야 하는지”에 더 집중하는 설정입니다. 필요하면 `completion_only_loss=False`로 바꿔 전체 시퀀스에 loss를 계산할 수도 있습니다. ([Hugging Face][2])

넷째, **Colab에서는 LoRA/PEFT를 같이 쓰는 것이 실용적**이라는 점입니다.
TRL SFT 문서는 PEFT 통합을 지원한다고 설명하며, adapter 학습 시에는 보통 더 높은 learning rate, 예를 들어 약 `1e-4` 수준을 많이 사용한다고 안내합니다. ([Hugging Face][2])

---

# 4. Module 4 실습 예제

## Colab에서 바로 실행하는 SFTTrainer 예제

아래 예제는 **Colab 단일 GPU 기준**의 교육용 SFT 실습입니다.
전제는 `/content/module3_sft_dataset.jsonl` 파일이 이미 존재한다는 것입니다.

---

## Cell 1. 패키지 설치

```python
!pip -q install -U transformers datasets trl peft accelerate sentencepiece
```

---

## Cell 2. 기본 import

```python
import os
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
```

---

## Cell 3. 실험 설정

```python
set_seed(42)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_PATH = "/content/module3_sft_dataset.jsonl"
OUTPUT_DIR = "/content/module4_sft_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("MODEL_NAME:", MODEL_NAME)
print("DATA_PATH:", DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Cell 4. 데이터셋 로드

Hugging Face `datasets`는 로컬 JSON 파일을 `load_dataset("json", data_files=...)` 형태로 로드할 수 있습니다. ([Hugging Face][4])

```python
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(dataset)
print(dataset[0])
```

---

## Cell 5. train / eval 분리

```python
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("train size:", len(train_dataset))
print("eval size :", len(eval_dataset))
```

---

## Cell 6. 샘플 확인

```python
for i in range(min(3, len(train_dataset))):
    print("=" * 80)
    print("PROMPT:")
    print(train_dataset[i]["prompt"])
    print("\nCOMPLETION:")
    print(train_dataset[i]["completion"])
```

---

## Cell 7. 토크나이저 / 모델 로드

SFT 문서에 따르면 `processing_class`로 tokenizer를 넘길 수 있고, padding token이 필요합니다. pad token이 없으면 eos token을 fallback으로 쓸 수 있습니다. ([Hugging Face][2])

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype
)

model.config.use_cache = False

print("Loaded model and tokenizer.")
print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)
```

---

## Cell 8. LoRA 설정

TRL SFT 문서는 PEFT 통합을 지원하며, `peft_config=LoraConfig()` 형태의 예제를 제공합니다. adapter 학습 시에는 보통 learning rate를 더 높게 잡는 편이라고도 설명합니다. ([Hugging Face][2])

```python
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear"
)
```

---

## Cell 9. SFTConfig 설정

`prompt-completion` 데이터셋에서는 기본적으로 completion 부분에만 loss가 계산됩니다. 교육용 예제에서는 그 의도를 분명히 보여주기 위해 `completion_only_loss=True`를 명시해 두는 편이 좋습니다. `SFTConfig`는 `max_length`, `packing`, `processing_class`, `eval_strategy` 같은 설정을 함께 제어할 수 있습니다. ([Hugging Face][2])

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    max_length=512,
    packing=False,
    completion_only_loss=True,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
)
```

---

## Cell 10. SFTTrainer 생성

SFT quickstart는 아주 짧은 형태로 `SFTTrainer(model=..., train_dataset=...)`를 보여주고, SFT 문서는 prompt-completion 데이터를 그대로 trainer에 전달할 수 있다고 설명합니다. ([Hugging Face][1])

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

---

## Cell 11. 학습 전 샘플 생성 함수

```python
def generate_text(model, tokenizer, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
```

---

## Cell 12. 학습 전 baseline 확인

```python
test_prompts = [
    "27 * 14 의 결과만 답하세요.",
    "name, role 키를 가진 JSON만 출력하세요. name은 Alice, role은 engineer.",
    "고객 문의에 정중한 한국어로 3문장 이내로 답하세요: 배송이 늦어지고 있습니다."
]

print("=== BEFORE SFT ===")
for p in test_prompts:
    print("=" * 80)
    print("PROMPT:", p)
    print("OUTPUT:", generate_text(model, tokenizer, p))
```

---

## Cell 13. 학습 실행

```python
train_result = trainer.train()
print(train_result)
```

---

## Cell 14. 어댑터 및 체크포인트 저장

```python
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to:", OUTPUT_DIR)
```

---

## Cell 15. 학습 후 출력 비교

```python
trained_model = trainer.model

print("=== AFTER SFT ===")
for p in test_prompts:
    print("=" * 80)
    print("PROMPT:", p)
    print("OUTPUT:", generate_text(trained_model, tokenizer, p))
```

---

## Cell 16. 간단 관찰 메모 저장

```python
notes = """# Module 4 SFT Observation

## What changed after SFT?
- 

## Which category improved the most?
- 

## Which category still needs more work?
- 

## Is SFT enough for this task?
- 

## Candidate next step
- DPO / PPO / GRPO
- reason:
"""

with open("/content/module4_sft_observation.md", "w", encoding="utf-8") as f:
    f.write(notes)

print("Saved: /content/module4_sft_observation.md")
```

---

# 5. 이 예제에서 강의자가 꼭 설명해야 할 것

이 예제의 핵심은 “코드가 돌아간다”보다 **왜 이렇게 설정했는가**입니다.

첫째, `module3_sft_dataset.jsonl`을 그대로 쓸 수 있는 이유입니다.
`SFTTrainer`는 공식적으로 `prompt-completion` 형식을 지원하므로, Module 3에서 만든 데이터셋이 바로 다음 단계 학습 입력으로 이어집니다. ([Hugging Face][2])

둘째, `completion_only_loss=True`의 의미입니다.
이번 실습은 “질문을 재현하는 법”보다 “답변을 원하는 방향으로 생성하는 법”을 배우는 것이므로 completion-only가 교육적으로 잘 맞습니다. 공식 docs도 prompt-completion 데이터셋에서는 completion 부분에만 loss를 계산하는 것이 기본 동작이라고 설명합니다. ([Hugging Face][2])

셋째, LoRA를 쓴 이유입니다.
공식 SFT 문서는 PEFT 통합을 지원한다고 설명하고, adapter 학습을 사용할 때는 전체 모델을 전부 업데이트하는 것보다 실습 부담이 줄어드는 구성이 가능하다는 점을 보여줍니다. 그래서 Colab에서는 LoRA 기반 예제가 훨씬 현실적입니다. ([Hugging Face][2])

넷째, SFT의 한계입니다.
SFT는 “좋은 정답 예시를 따라 하게 만드는 것”에는 매우 좋지만, “둘 중 어느 답이 더 선호되는가”나 “보상 최적화”까지 직접 해결하는 방식은 아닙니다. 그래서 다음 단계에서 DPO, PPO, GRPO로 넘어가는 흐름이 자연스럽습니다. TRL quickstart도 SFT, GRPO, DPO를 별도 trainer 흐름으로 분리해 보여 줍니다. ([Hugging Face][1])

# 6. 실습 과제 템플릿

과제는 간단하게 아래처럼 주면 좋습니다.

과제 제목:
**`module3_sft_dataset.jsonl`로 SFT를 수행하고 전후 출력을 비교하라**

제출물:

* 학습 로그 캡처
* 학습 전후 출력 비교 3건
* `module4_sft_observation.md`

관찰 질문:

* 가장 크게 좋아진 범주는 무엇인가?
* 여전히 약한 범주는 무엇인가?
* 이 문제는 SFT만으로 충분한가?
* 다음 단계로 DPO, PPO, GRPO 중 무엇이 적합한가?

# 7. 다음 모듈 연결

이 Module 4가 끝나면 가장 자연스러운 다음 단계는 두 갈래입니다.

* **Module 5**에서 SFT 결과를 평가 루브릭으로 정리하고 baseline과 비교
* **Module 6**에서 같은 과제를 `module3_dpo_dataset.jsonl`로 DPO 실습으로 확장

원하시면 다음 답변에서 이 내용을 바로 이어서 **Module 4용 Colab notebook 완성본(.ipynb) 구조**로 정리해 드리겠습니다.

[1]: https://huggingface.co/docs/trl/quickstart "Quickstart · Hugging Face"
[2]: https://huggingface.co/docs/trl/sft_trainer "SFT Trainer · Hugging Face"
[3]: https://huggingface.co/docs/trl/main/dataset_formats "Dataset formats and types · Hugging Face"
[4]: https://huggingface.co/docs/datasets/en/package_reference/loading_methods?utm_source=chatgpt.com "Loading methods · Hugging Face"


