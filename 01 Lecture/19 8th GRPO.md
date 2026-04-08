GRPO 모듈도 앞선 PPO/DPO 모듈과 같은 형식으로 강의 교안 + Colab notebook을 함께 묶겠습니다. 이번에는 특히 DeepSeekMath의 GRPO 설명과 현재 TRL의 GRPOTrainer 입력 형식·reward 함수·로그 항목을 기준으로, PPO와 같은 조건에서 비교 실습이 되도록 구성



작성해 두었습니다.

다운로드:
[Module 8 GRPO 강의 교안 (.md)](sandbox:/mnt/data/module8_grpo_teacher_guide.md)
[Module 8 GRPO Colab notebook 완성본 (.ipynb)](sandbox:/mnt/data/module8_grpo_colab_complete.ipynb)

이번 구성은 **DeepSeekMath의 GRPO 설명**과 **현재 TRL의 GRPOTrainer 문서**를 기준으로 정리했습니다. DeepSeekMath는 GRPO를 **PPO의 변형**으로 소개하며, 수학 추론을 강화하면서도 **PPO의 메모리 사용을 줄이는 방향**을 강조합니다. TRL 문서도 GRPO를 **online learning algorithm**으로 설명하고, prompt마다 여러 completion을 생성해 reward를 계산한 뒤 **group-relative advantage**로 학습한다고 안내합니다. 또한 현재 TRL 문서에서는 **`beta=0.0`이 기본**이라 KL 항을 기본적으로 사용하지 않는 설정을 제시합니다. ([arXiv][1])

노트북에는 다음이 포함되어 있습니다.

* `module3_grpo_dataset.jsonl` 로드
* `module4_sft_output` 우선 사용, 없으면 base instruct model fallback
* PPO와 같은 철학의 **rule-based reward 함수** 구성
* `GRPOTrainer` 실행
* **SFT vs GRPO** before/after 비교
* reward 기반 scorecard 저장
* 가능하면 **Module 7 PPO 요약 파일**을 읽어 **PPO vs GRPO 비교 메모**까지 이어지게 하는 흐름

TRL 문서상 GRPO의 custom reward function은 `prompts`, `completions`, `completion_ids`와 데이터셋의 추가 컬럼을 입력으로 받을 수 있고, train dataset에는 최소 `"prompt"` 컬럼이 필요합니다. ([Hugging Face][2])

다음 단계는 자연스럽게 **SFT vs DPO vs PPO vs GRPO 통합 비교 모듈**입니다.

[1]: https://arxiv.org/abs/2402.03300 "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
[2]: https://huggingface.co/docs/trl/grpo_trainer "GRPO Trainer · Hugging Face"

