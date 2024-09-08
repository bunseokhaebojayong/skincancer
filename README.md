## 실전경진대회 ISIC 2024 경진대회 학습 코드

## 팀 소개
안녕하세요. 저희는 분석해보자옹 팀입니다!
ISIC 2024 대회를 준비하면서 저희가 학습 서버에서 시도해봤던 흔적들입니다.

- best_record.csv : Epoch 별로 가장 좋은 auc와 loss 결과를 모델별로 기록해 둔 csv 파일입니다
- check_transferlearning_timm.py : 모델 확인 코드입니다.
- concat1-save-csv-file : training한 모델의 결과를 메타데이터에 추가한 결과입니다
- concat2-cncat-custom-data-and-meta-dataipynb : 메타데이터에 추가한 결과를 머신러닝으로 추론한 결과입니다.
- data_analysis.ipynb : EDA한 ipynb 파일입니다.
- history.csv : 모델별로 훈련 결과를 기록해둔 파일입니다
- hjdeepsad.ipynb : deepsad 관련 실험을 수행한 ipynb 파일입니다.
- inference.py - valid 데이터셋 inference하는 파이썬 코드입니다.
- model.py : 모델 클래스 코드입니다
- multi_inference.py : 멀티gpu 및 양자화 관련해서 파이썬으로 작성된 추론코드입니다.
- multi_run.sh : 멀티gpu를 사용해서 훈련하는 쉘 스크립트 파일입니다.
- multi_train.py : 멀티gpu 및 양자화 실험을 위해 작성된 파이썬 코드입니다.
- multi_utils.py : Data Agumentation + 데이터셋 구축 + 데이터로더에 멀티gpu 사용을 위해 데이터로더 파트를 좀 수정한 파이썬 코드입니다.
- multiquant_model.py : 양자화 실험을 시도해본 파이썬 코드입니다.
- pooling.py : GeM Pooling을 적용한 코드인데, ViT와 같은 몇몇 모델에 적용해야 하는 경우 ViTGem이라는 클래스로 별도로 정의하고, 코드의 차이가 있습니다.
- run.sh : 일반 train 쉘 스크립트 파일입니다.(단일gpu 사용시)
- solution-with-transformer.ipynb : ViT 실험 코드입니다.
- train.py : 단일 gpu 사용시 사용하는 훈련 파이썬 코드입니다.
- utils.py : Data Augmentation + 데이터셋 구축 + 데이터로더를 위한 파이썬 코드입니다.
