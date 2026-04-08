Module 9. 통합 비교 프로젝트: SFT → DPO / PPO / GRPO용 강의 교안 완성본입니다.
이 모듈은 앞선 실습 결과를 하나의 비교 프레임으로 수렴시키는 마지막 단계입니다. TRL quickstart는 SFT, GRPO, DPO를 대표적인 quick example로 제시하고, 각 예제가 compact model 기준의 빠른 실험을 목표로 한다고 설명합니다. DPO 문서는 preference data 기반 정렬, PPO 문서는 reward 신호 기반 RL, GRPO 문서는 PPO의 변형이자 group-relative advantage 기반 학습으로 설명합니다.


아래는 **Module 9. 통합 비교 프로젝트: SFT → DPO / PPO / GRPO**용 **강의 교안 완성본**입니다.
이 모듈은 앞선 실습 결과를 하나의 비교 프레임으로 수렴시키는 마지막 단계입니다. TRL quickstart는 SFT, GRPO, DPO를 대표적인 quick example로 제시하고, 각 예제가 compact model 기준의 빠른 실험을 목표로 한다고 설명합니다. DPO 문서는 preference data 기반 정렬, PPO 문서는 reward 신호 기반 RL, GRPO 문서는 PPO의 변형이자 group-relative advantage 기반 학습으로 설명합니다. ([Hugging Face][1])

---

# Module 9. 통합 비교 프로젝트

## DPO vs PPO vs GRPO

## 1. 모듈 목표

이 모듈의 목표는 세 방법을 각각 잘 돌려보는 데서 끝나지 않습니다.
핵심은 **같은 base model, 같은 평가셋, 비슷한 과제 조건**에서 `SFT → DPO`, `SFT → PPO`, `SFT → GRPO`를 나란히 비교하고, 결과를 **데이터 준비 난이도, 구현 복잡도, 계산 자원, 최종 성능**의 네 축으로 정리하는 것입니다. DPO는 preference data만으로 직접 최적화하는 반면, PPO는 reward/value 기반 온라인 RL이고, GRPO는 PPO의 변형으로 수학 추론 강화와 메모리 사용 최적화를 강조합니다. ([Hugging Face][2])

## 2. 학습 성과

수강생은 모듈 종료 시 아래를 설명할 수 있어야 합니다.

* **DPO**는 왜 “offline preference alignment”라고 부를 수 있는가. DPO 문서는 preference dataset의 positive/negative pair를 수집한 뒤, DPO loss를 직접 최대화하는 2단계 흐름을 제시합니다. ([Hugging Face][2])
* **PPO**는 왜 reward/value 구조와 온라인 샘플링이 필요한가. PPOTrainer 문서는 reward signal을 handcrafted rule, metric, reward model 등에서 받을 수 있다고 안내하며, RLHF 계열 참고 문헌을 함께 제시합니다. ([Hugging Face][3])
* **GRPO**는 왜 PPO 계열이지만 더 가벼운 RL 실험 후보로 볼 수 있는가. GRPO 문서는 DeepSeekMath를 근거로 PPO의 변형이며 memory usage를 함께 최적화한다고 설명하고, 현재 구현에서는 `beta=0.0`을 기본으로 둔다고 명시합니다. ([Hugging Face][4])
* 어떤 과제는 DPO가 유리하고, 어떤 과제는 PPO/GRPO가 더 직접적인 최적화를 제공하는지 말할 수 있어야 합니다. ([Hugging Face][2])

---

## 3. 모듈 구성

### 전체 진행 흐름

1. **공통 실험 조건 고정**
2. **트랙 선택**
3. **세 경로 실행**
4. **평가셋으로 비교**
5. **통합 비교표 작성**
6. **최종 보고서 발표**

이 흐름의 핵심은 “모델만 다르고 나머지는 최대한 동일하게”가 아니라, **출발점과 평가 조건을 동일하게 유지**하는 것입니다. 그래야 방법 차이와 실험 조건 차이를 구분할 수 있습니다.

---

## 4. 공통 실험 조건

모든 조는 아래 조건을 고정합니다.

* 같은 **base model**
* 같은 **SFT 초기 체크포인트**
* 같은 **평가셋**
* 같은 **generation config**
* 같은 **비교 항목 4개**

  * 데이터 준비 난이도
  * 구현 복잡도
  * 계산 자원
  * 최종 성능

이렇게 해야 DPO, PPO, GRPO의 차이가 **방법론 차이**로 해석됩니다.

---

## 5. 두 개의 프로젝트 트랙

## Track A. Persona Alignment Track

목표는 **톤, 친절성, 일관성, 응답 선호도**를 비교하는 것입니다.
이 트랙에서는 DPO가 특히 잘 드러납니다. DPO 문서는 human preference pair 기반 정렬을 위한 trainer라고 설명하고, 원 논문 요약에서도 RLHF보다 단순하면서도 preference alignment를 수행할 수 있다고 소개합니다. ([Hugging Face][2])

### 권장 과제 예시

* 더 정중한 고객 응대
* 더 짧고 친절한 답변
* 더 안전한 거절 응답
* 같은 의미를 유지하면서 더 일관된 assistant persona 만들기

### 비교 포인트

* DPO가 chosen/rejected 차이를 얼마나 잘 반영하는가
* PPO/GRPO가 reward로 공손성·길이·안전성을 얼마나 안정적으로 밀어올리는가
* SFT만으로 충분했던 부분과 DPO가 특히 강했던 부분은 무엇인가

---

## Track B. Capability Improvement Track

목표는 **정답률, structured output 성공률, reasoning/accuracy reward**를 비교하는 것입니다.
이 트랙에서는 PPO와 GRPO가 특히 잘 드러납니다. PPO는 reward signal 기반 온라인 RL로 설계되어 있고, GRPO는 같은 계열이지만 DeepSeekMath와 TRL 문서에서 더 경량화된 RL 실험 흐름으로 제시됩니다. ([Hugging Face][3])

### 권장 과제 예시

* 간단한 수학 정답률 향상
* JSON 출력 형식 준수율 향상
* required keys를 가진 structured output 성공률 향상
* reasoning-style 답변의 정확도 개선

### 비교 포인트

* 명시적 reward가 있을 때 PPO가 얼마나 직접적으로 최적화되는가
* 같은 reward 함수에서 GRPO가 PPO 대비 얼마나 가볍고 안정적인가
* value/critic 의존성이 줄어든 것이 실제 운영에 어떤 장단점을 주는가

---

## 6. 방법별 비교 프레임

| 구분            | DPO                             | PPO                | GRPO                             |
| ------------- | ------------------------------- | ------------------ | -------------------------------- |
| 핵심 데이터        | preference pair                 | prompt + reward    | prompt + reward                  |
| 학습 형태         | offline preference optimization | online RL          | online RL                        |
| 대표 특징         | reward model 없이 직접 정렬           | reward/value 구조 포함 | PPO 변형, group-relative advantage |
| 강점이 잘 드러나는 과제 | 톤, 선호, 스타일                      | 정답/형식 보상 최적화       | reasoning/accuracy 중심 보상         |
| 실습 난이도        | 상대적으로 낮음                        | 높음                 | 중간~높음                            |

이 표는 DPOTrainer, PPOTrainer, GRPOTrainer 문서와 DeepSeekMath 설명을 교육용 비교축으로 단순화한 것입니다. DPO는 preference data, PPO는 reward signal 기반, GRPO는 PPO의 variant로 소개됩니다. ([Hugging Face][2])

---

## 7. 강의 진행안

### 1단계. 오프닝

강사는 먼저 세 문장을 칠판에 적습니다.

* DPO = **선호 비교를 직접 학습**
* PPO = **보상을 최대화하는 온라인 RL**
* GRPO = **PPO 계열의 더 가벼운 RL 실험 후보**

이 세 문장은 각각 DPO, PPO, GRPO 공식 문서와 DeepSeekMath 설명으로 뒷받침됩니다. ([Hugging Face][2])

### 2단계. 팀별 트랙 선택

각 조는 Persona Alignment 또는 Capability Improvement 중 하나를 선택합니다.

### 3단계. 세 경로 실행

모든 조는 아래 세 경로를 모두 실행합니다.

* 경로 A: `SFT → DPO`
* 경로 B: `SFT → PPO`
* 경로 C: `SFT → GRPO`

### 4단계. 동일 평가셋으로 비교

같은 프롬프트 세트, 같은 generation config, 같은 루브릭으로 비교합니다.

### 5단계. 보고서 작성

결과를 네 축으로 정리합니다.

* 데이터 준비 난이도
* 구현 복잡도
* 계산 자원
* 최종 성능

---

## 8. 평가셋 구성 가이드

### Persona Alignment Track 평가 항목

* 정중성
* 친절성
* 간결성
* 톤 일관성
* 응답 선호도

여기서는 **chosen 응답과 유사한 출력**을 얼마나 잘 만드는지가 중요합니다. 따라서 DPO의 효과가 잘 드러나는 설계가 됩니다. ([Hugging Face][2])

### Capability Improvement Track 평가 항목

* 수학 정답률
* JSON parse 성공률
* required keys 충족률
* 정답 형식 일치율
* reward 평균

여기서는 reward를 정의할 수 있어야 하므로 PPO/GRPO가 훨씬 자연스럽습니다. PPO 문서는 reward signal을 handcrafted rule, metric, reward model 등으로 둘 수 있다고 안내하고, GRPO 문서도 reward function 중심 실험을 바로 보여 줍니다. ([Hugging Face][3])

---

## 9. 조별 제출물

### 필수 제출물

1. **비교표 1장**
2. **실험 보고서**
3. **전후 출력 비교 샘플 6건 이상**
4. **최종 발표 슬라이드 5장 내외**

### 비교표 필수 항목

* DPO 데이터 준비 시간
* PPO reward 설계 시간
* GRPO reward 설계 시간
* 학습 시간
* GPU 메모리 사용량
* 가장 좋아진 항목
* 가장 나빠진 항목
* 실패 사례

---

## 10. 실험 보고서 템플릿

## 10-1. 프로젝트 개요

* 트랙:
* base model:
* SFT checkpoint:
* 평가셋 설명:

## 10-2. 경로별 설정

### A. SFT → DPO

* dataset:
* chosen/rejected 기준:
* 장점:
* 어려웠던 점:

### B. SFT → PPO

* reward 함수:
* 로그에서 본 핵심 지표:
* 장점:
* 어려웠던 점:

### C. SFT → GRPO

* reward 함수:
* group-relative setup:
* 장점:
* 어려웠던 점:

## 10-3. 통합 비교

* 데이터 준비 난이도:
* 구현 복잡도:
* 계산 자원:
* 최종 성능:

## 10-4. 최종 결론

* Persona 과제에는 무엇이 더 적합했는가?
* Capability 과제에는 무엇이 더 적합했는가?
* 실제 프로젝트라면 어떤 순서로 적용할 것인가?

---

## 11. 강사용 해설 포인트

이 모듈의 핵심은 “누가 더 좋다”가 아닙니다.
오히려 아래 질문에 답하게 만드는 것이 중요합니다.

**왜 이 과제에서는 DPO가 잘 맞고, 다른 과제에서는 PPO/GRPO가 더 직접적이었는가?**

강사가 강조해야 할 포인트는 세 가지입니다.

첫째, **DPO는 데이터 준비가 비교적 직관적**입니다.
선호쌍만 잘 만들면 되기 때문입니다. DPO 문서도 preference dataset 수집과 직접 loss 최적화를 핵심 단계로 제시합니다. ([Hugging Face][2])

둘째, **PPO는 더 무겁지만 reward가 분명한 문제에 강합니다.**
PPO 문서는 reward signal을 명시적으로 받아 RL을 수행하는 구조를 전제로 합니다. 따라서 수학 정답, 형식 준수, 안전 점수처럼 채점이 명확한 문제에 잘 맞습니다. ([Hugging Face][3])

셋째, **GRPO는 PPO와 최대한 같은 조건으로 비교해야 의미가 있습니다.**
GRPO 문서와 DeepSeekMath는 PPO의 variant이자 memory 효율을 강조하므로, 같은 prompt 세트와 같은 reward 함수에서 비교해야 장단점이 드러납니다. 현재 TRL 문서가 `beta=0.0`을 기본으로 둔다는 점도 PPO와 운영 철학이 완전히 같지는 않다는 신호입니다. ([Hugging Face][4])

---

## 12. 발표 질문 예시

발표 때 강사가 던지기 좋은 질문입니다.

* DPO는 어떤 선호 기준에서 가장 효과가 컸나요?
* PPO reward는 설계는 쉬웠지만 학습은 어려웠나요?
* GRPO는 PPO보다 실제로 가벼웠나요?
* 최종 성능만 보면 누가 좋았고, 운영 현실까지 고려하면 누가 좋았나요?
* “우리 팀 과제”에서는 어떤 순서가 가장 합리적이었나요?
  예: `SFT → DPO`, 또는 `SFT → PPO`, 또는 `SFT → GRPO`

---

## 13. 최종 정리 문구

이 모듈이 끝나면 수강생은 아래처럼 정리할 수 있어야 합니다.

* **DPO는 offline preference alignment에 강하다.** preference pair만으로도 스타일, 톤, 선호 정렬을 꽤 직접적으로 만들 수 있다. ([Hugging Face][2])
* **PPO는 reward/value 기반 online RL이다.** 구현은 복잡하지만 reward가 명확한 과제에서는 가장 직접적인 최적화 수단이 될 수 있다. ([Hugging Face][3])
* **GRPO는 PPO 계열이지만 더 가벼운 RL 실험 후보다.** 특히 reasoning/accuracy 중심 reward에서 교육적으로 비교 가치가 크다. ([Hugging Face][4])

---

원하시면 이 교안을 이어서 **Module 9 통합 비교용 Colab notebook** 구조까지 바로 만들어 드리겠습니다.

[1]: https://huggingface.co/docs/trl/quickstart "Quickstart · Hugging Face"
[2]: https://huggingface.co/docs/trl/v0.22.2/dpo_trainer "DPO Trainer · Hugging Face"
[3]: https://huggingface.co/docs/trl/ppo_trainer "PPO Trainer · Hugging Face"
[4]: https://huggingface.co/docs/trl/grpo_trainer "GRPO Trainer · Hugging Face"




