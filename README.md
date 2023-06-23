# VAEP-KCC

-StatsBomb데이터를 활용하여 모든 플레이별 가치를 평가하는 지표 :VAEP
-VAEP는 각 플레이의 득점 및 실점 확률을 예측하여 득점확률변화량-실점확률변화량로 계산된다.
-feature : 행동 유형, 행동 결과, 위치, 골까지 거리, 골까지 각도등 활용
-label : 각 플레이 이후 10개의 플레이 안에 득점을 하면 score_label = 1, 실점을 하면 concede_label=1

-기존에는 boosting, random-forest, logistic등의 Machine learning기법을 활용하여 득점 및 실점 확률을 예측했지만
\n-본 연구에서는 deep-learning기법을 활용하여 예측함으로써 Machine learning기법보다 성능 향상을 검증함

1.soccer data load.ipynb : StatsBomb데이터를 loading하는 작업
-본 연구에서는 LaLiga데이터를 활용할 예정
-game_id = 3,773,689는 특정 플레이 이후 score_label=1, concede_label=1인 데이터가 존재함. 이는 부정확한 데이터이므로 해당 경기 데이터는 제거함
