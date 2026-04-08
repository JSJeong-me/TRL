8개 module을 **“SFT → DPO → PPO → GRPO를 모두 실습하고, 특히 PPO·DPO·GRPO의 차이를 직접 비교해서 이해하는 과정”**으로 다시 설계합니다.

DPO는 **선호쌍(preference pair)** 을 이용해 분류형 목적함수로 정렬하는 방식이고, PPO는 **롤아웃-보상-정책 업데이트**를 반복하는 온라인 RL 방식이며, GRPO는 **PPO의 변형**으로 소개되며 그룹 상대 비교를 통해 PPO 대비 메모리 사용을 줄이는 방향으로 설명됩니다. 현재 TRL도 SFT, DPO, GRPO를 빠른 시작 예제로 제시하고 있으며, PPO 역시 별도 Trainer로 지원합니다. 또한 TRL의 데이터 형식 가이드에 따르면 DPO는 preference 데이터, GRPO는 prompt-only 데이터, PPO는 tokenized language modeling 계열 입력을 기대합니다. ([arXiv][1])

아래 재설계안의 핵심은, **같은 base model과 비슷한 과제를 두고 세 방법을 나란히 실험**하게 만드는 것입니다.
즉 수강생이 단순히 정의를 외우는 것이 아니라, “어떤 데이터가 필요한가”, “보상모델·가치모델이 필요한가”, “학습이 얼마나 불안정한가”, “어느 과제에서 어떤 방법이 더 잘 맞는가”를 직접 체감하도록 구성합니다. 이 비교축은 각 module에 누적되며 마지막 Module 8에서 종합 비교로 마무리합니다. ([Hugging Face][2])

## Module 1. Post-Training 전체 지도와 PPO·DPO·GRPO 위치 이해

이 모듈에서는 포스트 트레이닝의 큰 그림을 먼저 잡습니다.
SFT는 기본 능력과 스타일을 주입하는 출발점, DPO는 **오프라인 preference pair 기반 정렬**, PPO는 **온라인 RL 기반 정책 최적화**, GRPO는 **PPO 계열의 경량화/메모리 절감형 변형**으로 배치합니다. 수강생이 이 단계에서 최소한 “왜 DPO는 선호데이터가 필요하고, 왜 PPO/GRPO는 보상 중심 사고가 필요한지”를 말할 수 있어야 합니다. ([arXiv][1])

실습은 아주 가볍게 시작합니다.
같은 질문에 대해 “좋은 답 / 나쁜 답”을 직접 라벨링해 보고, 또 같은 질문에 대해 “보상 함수로 평가 가능한 답”이 무엇인지 분리해 보게 합니다. 여기서부터 수강생은 **DPO형 데이터 사고**와 **RL형 데이터 사고**가 다르다는 점을 체감합니다. ([Hugging Face][2])

## Module 2. SmolLM 실습 환경 구축과 Baseline 측정

이 모듈은 실제 실습 기반을 만드는 단계입니다.
SmolLM 또는 동급 소형 모델을 내려받아 baseline inference를 수행하고, 이후 모든 실험이 동일한 출발선에서 비교되도록 구성합니다. TRL은 소형 모델을 사용한 SFT, DPO, GRPO quick example을 제공하고 있으므로, 교육용 실습도 같은 철학으로 **작은 모델에서 반복 가능한 실험**으로 설계하는 것이 적절합니다. ([Hugging Face][3])

실습에서는 동일한 테스트셋으로 baseline을 먼저 측정합니다.
예를 들어 두 개의 공통 과제를 둡니다. 하나는 **persona/style 정렬 과제**, 다른 하나는 **수학 또는 포맷 준수 과제**입니다. 이후 SFT, DPO, PPO, GRPO의 모든 결과는 이 baseline과 비교합니다. 이 모듈의 산출물은 “튜닝 전 모델 응답집”입니다.

## Module 3. 데이터 큐레이션: SFT용, DPO용, PPO/GRPO용 데이터 분리 설계

이 모듈은 이번 재설계의 핵심입니다.
TRL 문서 기준으로 DPO는 `prompt/chosen/rejected` 형태의 preference 데이터가 핵심이고, GRPO는 prompt-only 데이터로도 학습 가능하며, PPO는 reward/value 기반 RL 루프에 맞는 입력 구성이 필요합니다. 따라서 이번 강의에서는 하나의 원천 과제를 세 갈래 데이터셋으로 변환하는 연습을 넣습니다. ([Hugging Face][2])

실습은 세 단계로 나눕니다.
먼저 SFT용 `instruction → response` 데이터를 만들고, 같은 샘플에서 DPO용 `prompt/chosen/rejected`를 만듭니다. 마지막으로 PPO/GRPO용으로는 **prompt + 자동 보상 계산 규칙**을 설계합니다. 이 과정을 통해 수강생은 “같은 문제라도 학습법이 달라지면 데이터셋 구조가 완전히 달라진다”는 점을 실감하게 됩니다. ([Hugging Face][2])

## Module 4. SFT 구현: 공통 출발점 만들기

이 모듈은 PPO·DPO·GRPO 비교를 위한 공통 기반 모델을 만드는 단계입니다.
먼저 SFT로 모델의 기본 instruction-following, persona, 출력 형식을 안정화합니다. 교육적으로는 이 SFT 모델을 이후 DPO/PPO/GRPO의 공통 초기 정책으로 사용해야 비교가 훨씬 명확해집니다. TRL은 SFTTrainer를 quickstart의 첫 단계로 제시합니다. ([Hugging Face][3])

실습은 두 트랙으로 나눕니다.
첫 번째는 **assistant 성격 바꾸기**이고, 두 번째는 **정답 형식 맞추기**입니다. 여기서 만든 SFT 체크포인트를 이후 세 가지 비교 실습의 공통 시작점으로 씁니다. 이 모듈의 핵심 메시지는 “비교는 같은 출발선에서 해야 한다”입니다.

## Module 6. DPO 실습: 오프라인 선호학습이 어디까지 가능한가

이 모듈은 DPO를 독립적으로 실습하는 단계입니다.
DPO는 RLHF의 목표를 분류형 손실로 직접 최적화하는 방식으로 소개되며, 논문은 이를 **stable, performant, computationally lightweight** 하다고 설명합니다. TRL의 DPOTrainer 역시 preference 데이터셋을 사용한 빠른 실험 구성을 제공합니다. ([arXiv][1])

실습은 같은 prompt에 대해 `chosen`과 `rejected`를 직접 설계해 보는 것에서 시작합니다.
예를 들면 “더 정중한 답”, “더 짧고 정확한 답”, “더 안전한 답”처럼 선호 기준을 바꿔 여러 DPO 데이터셋을 만들어 봅니다. 이후 SFT 모델과 DPO 모델을 비교하면서, DPO가 **보상모델을 따로 두지 않고도** 선호 정렬을 얼마나 만들 수 있는지 확인합니다. ([arXiv][1])

여기서 첫 번째 비교 실습을 넣습니다.
수강생은 동일한 평가셋에 대해 “SFT vs DPO”를 비교하고, DPO가 잘하는 부분과 못하는 부분을 적습니다. 보통 이 단계에서는 **스타일 정렬, 응답 선호도, 톤 일관성** 같은 항목이 잘 드러나도록 설계하는 것이 교육 효과가 좋습니다.

## Module 7. PPO 실습: 보상모델과 가치모델이 들어가는 온라인 RL

이 모듈에서는 PPO를 통해 “진짜 RLHF형 루프”를 경험하게 합니다.
PPO는 환경에서 샘플을 수집하고 surrogate objective로 여러 minibatch epoch를 수행하는 정책경사 계열 방법으로 제안되었고, TRL의 PPOTrainer 문서에서도 reward model과 value model, KL 관련 로그를 전제로 한 구조가 드러납니다. 즉 PPO는 DPO보다 구성요소가 더 많고, 학습 루프도 더 온라인적입니다. ([arXiv][4])

실습은 너무 무겁지 않게 설계합니다.
처음에는 복잡한 human preference 대신 **규칙 기반 reward** 또는 간단한 reward model로 시작합니다. 예를 들어 수학 문제에서는 정답 일치 여부를 점수로 쓰고, 포맷 과제에서는 JSON 형식 준수 여부를 점수로 씁니다. 그 다음 PPO 로그에서 `objective/kl`, `objective/scores`, `loss/value_avg` 같은 지표를 읽게 하여, PPO가 단순 지도학습이 아니라 **보상을 받아 정책과 가치 추정을 함께 다루는 과정**임을 이해하게 합니다. ([Hugging Face][5])

이 모듈의 비교 포인트는 명확합니다.
DPO는 오프라인 preference pair만으로 진행되지만, PPO는 reward/value 구조와 온라인 샘플링이 필요합니다. 그래서 구현 복잡도와 계산비용은 커지지만, **명시적 reward가 잘 정의되는 과제**에서는 더 직접적인 최적화가 가능합니다. ([arXiv][1])

## Module 8. GRPO 실습: PPO와 무엇이 같고 무엇이 다른가

이 모듈은 PPO와 GRPO를 정면 비교하는 단계입니다.
DeepSeekMath는 GRPO를 **PPO의 변형**으로 소개하면서 수학 추론을 강화하면서도 PPO의 메모리 사용을 줄이는 방향을 강조합니다. TRL 문서도 GRPO를 prompt-only 데이터와 reward function 중심으로 빠르게 실험할 수 있게 안내하고 있습니다. 또한 GRPO Trainer 설명은 최근 구현에서 KL term 기본값을 0으로 두는 점과, 손실 설계가 PPO와 유사하지만 운영상 선택지가 다름을 보여 줍니다. ([arXiv][6])

실습은 PPO와 가능한 한 동일 조건으로 맞춥니다.
즉 같은 SFT 초기모델, 같은 prompt 세트, 같은 reward 함수를 사용하고, PPO 실험과 GRPO 실험을 따로 돌립니다. 수강생은 그 후 **학습 안정성, GPU 메모리 사용량, 구현 난이도, 응답 품질, 보상 상승 속도**를 비교 기록합니다. 이 실습의 목적은 “GRPO가 PPO보다 무조건 낫다”가 아니라, **critic/value-model 의존성이 줄어들었을 때 어떤 실무적 장단점이 생기는가**를 이해하게 하는 데 있습니다. ([arXiv][6])

이 모듈이 끝나면 수강생은 최소한 다음을 설명할 수 있어야 합니다.
“DPO는 offline preference alignment”, “PPO는 reward/value 기반 online RL”, “GRPO는 PPO 계열이지만 그룹 상대 비교를 활용해 보다 경량화된 RL 실험에 적합할 수 있다.” ([arXiv][1])

## Module 9. 통합 비교 프로젝트: DPO vs PPO vs GRPO

마지막 모듈은 세 방법을 나란히 비교하는 종합 프로젝트입니다.
각 조는 동일한 base model과 동일한 평가셋으로 세 가지 경로를 모두 실행합니다. 경로 A는 **SFT → DPO**, 경로 B는 **SFT → PPO**, 경로 C는 **SFT → GRPO**입니다. 비교 항목은 반드시 네 가지로 고정합니다: 데이터 준비 난이도, 구현 복잡도, 계산 자원, 최종 성능. 이 설계는 앞선 모듈에서 학습한 차이를 하나의 표와 실험 보고서로 수렴시키는 역할을 합니다.

종합 실습 과제는 두 트랙이 좋습니다.
첫째는 **Persona Alignment Track**으로, 톤·친절성·일관성 같은 선호 정렬을 본격 비교합니다. 여기서는 DPO가 특히 잘 드러납니다. 둘째는 **Capability Improvement Track**으로, 수학 정답률이나 structured output 성공률을 기준으로 PPO와 GRPO의 장단점을 확인합니다. GRPO는 reasoning/accuracy reward와 잘 맞는 실험으로 배치하면 교육 효과가 큽니다. ([arXiv][6])

최종 산출물은 단순 모델 체크포인트가 아니라 비교 보고서입니다.
반드시 “어떤 과제에서 DPO가 유리했는지”, “언제 PPO가 과투자였는지”, “GRPO가 PPO 대비 실험 효율에서 어떤 차이를 보였는지”를 정리하게 합니다. 이 단계가 있어야 수강생이 방법론을 도구로 선택하는 감각을 갖게 됩니다.

## 이 재설계의 핵심 차이

이전 8개 module이 **SFT → DPO → Online RL**의 일반 흐름이었다면, 이번 버전은 처음부터 끝까지 **PPO·DPO·GRPO 비교 실습**이 중심축입니다.
따라서 강의의 초점도 “각 방법을 배우는 것”에서 “같은 문제를 다른 post-training 방법으로 풀어 보고, 차이를 해석하는 것”으로 이동합니다.

가장 중요한 실습 질문은 세 가지입니다.
첫째, **데이터가 다르면 방법 선택이 어떻게 달라지는가**. 둘째, **보상 정의가 쉬운 과제와 선호쌍 정의가 쉬운 과제는 무엇이 다른가**. 셋째, **PPO와 GRPO는 실제로 어느 정도의 실험 복잡도 차이를 보이는가**. 이 세 질문에 답하게 만들면 강의가 훨씬 실전형이 됩니다. ([Hugging Face][2])


[1]: https://arxiv.org/abs/2305.18290 "[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[2]: https://huggingface.co/docs/trl/en/dataset_formats "Dataset formats and types · Hugging Face"
[3]: https://huggingface.co/docs/trl/quickstart "Quickstart · Hugging Face"
[4]: https://arxiv.org/abs/1707.06347 "[1707.06347] Proximal Policy Optimization Algorithms"
[5]: https://huggingface.co/docs/trl/ppo_trainer "PPO Trainer · Hugging Face"
[6]: https://arxiv.org/abs/2402.03300 "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
