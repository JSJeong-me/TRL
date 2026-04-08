DPOTrainer의 현재 입력 형식과 SFT 모델 → DPO 미세조정 → SFT vs DPO 비교 흐름을 공식 문서 기준으로 확인한 뒤, 바로 실행 가능한 notebook 파일로 정리



작성해 두었습니다.

DPO는 원 논문에서 RLHF의 복잡한 reward-model + RL 파이프라인 대신, 선호쌍에 대한 직접 최적화로 접근하며 “stable, performant, and computationally lightweight”하다고 설명합니다. TRL의 DPOTrainer도 `prompt`, `chosen`, `rejected`를 포함하는 preference dataset을 기대하고, 현재 문서에서는 explicit prompt 형식을 권장합니다. 또 `ref_model=None`이면 초기 정책을 reference로 사용하도록 안내합니다. ([arXiv][1])

다운로드:
[Module 6 DPO 강의 교안 (.md)](sandbox:/mnt/data/module6_dpo_teacher_guide.md)
[Module 6 DPO Colab notebook 완성본 (.ipynb)](sandbox:/mnt/data/module6_dpo_colab_complete.ipynb)

이 노트북은 `module3_dpo_dataset.jsonl`과 `module4_sft_output`을 받아서, SFT 모델에서 시작해 DPO를 수행하고, 같은 평가 prompt에서 **SFT vs DPO**를 비교하도록 구성했습니다. TRL 문서상 DPO는 preference dataset을 사용하고, quickstart에서도 SFT·DPO 같은 compact-model 실험 흐름을 중심으로 소개됩니다. ([Hugging Face][2])

[1]: https://arxiv.org/abs/2305.18290 "[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[2]: https://huggingface.co/docs/trl/v0.22.2/dpo_trainer "DPO Trainer · Hugging Face"
