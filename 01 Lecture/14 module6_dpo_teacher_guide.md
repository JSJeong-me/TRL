# Module 6. DPO 실습: 오프라인 선호학습이 어디까지 가능한가

## 모듈 개요
이 모듈은 Direct Preference Optimization(DPO)를 독립적으로 실습하는 단계다. DPO는 RLHF의 목표를
선호쌍(preference pairs)에 대한 분류형 손실로 직접 최적화하는 방법으로 소개되며, 원 논문은 이를
stable, performant, computationally lightweight 하다고 설명한다.
Hugging Face TRL의 DPOTrainer는 `prompt`, `chosen`, `rejected`를 포함하는 preference dataset을 기대하며,
공식 문서는 explicit prompt 형식을 권장한다.

## 학습 목표
- `module3_dpo_dataset.jsonl`의 `prompt/chosen/rejected` 구조를 이해한다.
- 같은 prompt에 대해 선호 기준을 바꿔 chosen/rejected를 설계할 수 있다.
- Module 4에서 만든 SFT 모델을 시작점으로 DPO를 수행할 수 있다.
- SFT와 DPO의 출력을 비교해 스타일 정렬, 톤 일관성, 선호 반영 정도를 평가할 수 있다.

## 권장 시간
- 강의 50분
- 데모 20분
- 실습 70분
- 비교/토론 30분

## 핵심 개념
### 1) 왜 DPO인가
- RLHF는 보통 reward model과 RL 단계를 포함해 복잡하다.
- DPO는 별도 reward model 없이 선호 데이터를 직접 최적화한다.
- 따라서 교육용 실습에서 PPO보다 훨씬 빠르게 preference alignment를 체감하기 좋다.

### 2) DPO 데이터 형식
기본 형식은 아래와 같다.

```json
{"prompt": "고객 문의에 정중하게 답하세요.", "chosen": "안녕하세요. 불편을 드려 죄송합니다...", "rejected": "배송이 늦었습니다. 기다려 주세요."}
```

### 3) 어떤 선호 기준을 줄 것인가
- 더 정중한 답
- 더 짧고 정확한 답
- 더 안전한 답
- 더 일관된 말투
- 형식(JSON 등)을 더 잘 지키는 답

## 강의 슬라이드 초안
1. DPO의 문제 설정: preference pair
2. RLHF 대비 DPO의 위치
3. DPO 데이터셋 구조
4. explicit prompt를 권장하는 이유
5. SFT 모델을 먼저 쓰는 이유
6. 같은 prompt에 chosen/rejected 만들기
7. SFT vs DPO 비교 항목
8. 실습 안내

## 실습 흐름
1. Module 3의 `module3_dpo_dataset.jsonl` 불러오기
2. prompt / chosen / rejected 미리보기
3. Module 4의 SFT 산출물(`module4_sft_output`) 불러오기
4. SFT 모델로 평가 prompt에 대한 출력 생성
5. DPOTrainer로 학습
6. 같은 평가 prompt에 대해 DPO 모델 출력 생성
7. SFT vs DPO 비교표 저장
8. 관찰 메모 작성

## 평가 루브릭
### A. 스타일 정렬
- 공손한 표현 사용
- 공격적/딱딱한 어조 감소
- 말투 일관성

### B. 선호 반영
- chosen의 특성이 실제로 강화되었는가
- rejected의 경향이 줄었는가

### C. 과제 성능
- 형식 준수
- 안전 응답
- 짧고 명확한 답변

### D. 한계 파악
- DPO가 잘 못하는 과제는 무엇인가
- 단순 preference pair만으로 해결되지 않는 부분은 무엇인가

## 토론 질문
- chosen/rejected 차이가 너무 약하면 어떤 문제가 생길까?
- DPO는 어떤 과제에서 SFT보다 분명한 이점이 있는가?
- 정답 검증이 쉬운 수학 문제도 DPO가 적합한가, 아니면 PPO/GRPO가 더 적합한가?
