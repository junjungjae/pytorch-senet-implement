# pytorch-senet-implement

### Reference
- SeNet 공식 논문(HU, Jie; SHEN, Li; SUN, Gang. Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. p. 7132-7141.)
- se-resnet50 papers with code(https://paperswithcode.com/model/se-resnet?variant=seresnet50#)

### 이전 모델 구현 대비 개선점

- train, loss에 대한 loss, metric 저장 기능 추가(log & pkl)
- 기능별로 파일 분리
- 간단한 전처리 기능 추가

### 구현 & 학습결과

- 일반적으로 SENet은 독자적으로 쓰기보단 Block 모듈을 추가해서 쓴다고 해서 그렇게 사용해봄.
- MobileNet에 접목시켜 구현한 자료들은 많아 이전에 구현한 Resnet50에 접목시켜 구현
- 처음 테스트 때는 논문 구조 그대로(excitation -> fully connected layer) 구현
    1. benchmark 사이트(papers with code)에서 FC layer를 Conv2d로 사용했길래 사용
    2. 소요시간은 epoch당 약 6%(232s -> 247s) 정도 늘어남.
    3. 학습 최종결과는 train 99% / validation 71%(papers with code 기준 약 80위)
- 간단한 전처리(rotation, normalize)만 한것치곤 나쁘지 않은듯.

### 추후 시도 & 개선점

- 소규모 custom 데이터셋이라도 구해서 dataloader에 탑재하는 기능 추가
- 현재 imagenet 기준으로 정규화 수행. STL10에 맞게 정규화 수치 재조정 후 시도
- resnet, se-resnet, se-resnet(with conv2d exciation) 3개 결과 비교
- Dropout, EarlyStopCallback, LearningRateScheduler(ex: cosinedecay) 적용
- 이미지 전처리 기법 조사 & 적용
- 모델 학습 결과 시각화 좀더 다각도로...?
- Object detection 준비
