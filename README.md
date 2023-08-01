## level2_cv_semanticsegmentation-cv-17_sixseg

### Hand Bone Image Segmentation

##### - X-ray 이미지에서 사람의 뼈를 Segmentation 하는 인공지능 만들기

<img width="90%" src="https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-17/assets/70469008/dd3261a4-82d1-4424-974a-8f8f383d1158"/>

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

이번 프로젝트를 통해 만들어진 우수한 성능의 모델은 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용될 수 있을 것으로 기대됩니다. 🌎

#### Team Members

강대호
강정우
박혜나
서지훈
원유석
정대훈

#### 실험 진행 순서

1. EDA
2. pretrained 모델 성능 테스트
   - (SMP) DeepLabV3 / MAnet / PAN / PSP / FPN / UnetPlusPlus
   - (mmseg) upernet / segformer / segmenter / Mask2former / HRNet
3. EDA 기반 실험
4. data augmentation
   - VFlip / HFlip / RandomCrop / RandomGamma / RandomContrast / RandomBrightness / Blurring / Sharpenong / Scale / Shift / Rotation / Shearing
5. 모델 성능 개선
   - Loss funtion / K-fold / Resize / Train data relabeling
6. Ensemble

#### 최종 활용 모델

1. Unet++ (Resnet34)
2. Unet++ (Resnet152)
3. Unet++ (EfficientNetB5)
4. SegFormer
5. HRNet

#### Wrap-up Report

https://drive.google.com/file/d/1hN_A90BrdtJwqnJ7xcHaxt75cdXs7u5Y/view?usp=sharing

#### 평가 Metric

- Dice coefficient

#### Dataset

- number of images : 1100
  - train : 800
  - test : 300 (public 50% + private 50%)
  - 한 사람 당 2장의 이미지 존재 (왼손, 오른손)
- number of class : 크게 손가락 / 손등 / 팔로 구성되며, 총 29개의 뼈 종류(class)가 존재
- labels : 'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
  'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
  'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
  'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
  'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
  'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
- image size : (2048 x 2048), 3 channel

#### Input & Output

- Input
  - hand bone x-ray (png) : 한 사람 당 왼손, 오른손 총 2장의 이미지 제공
  - segmentation annotation (json) : segmentation mask가 points(polygon 좌표)로 제공
- Output
  - 각 pixel 좌표에 따른 class를 rle로 변환한 값 (csv)
