# VAEP(Valuing Actions by Estimating Probabilities)

<Introduction>
-An indicator that uses data called StatsBomb to evaluate the value of all plays : VAEP<br/>
-VAEP is calculated as the difference between the change in scoring probability and the change in conceding probability after predicting the score probability and the concede probability of each play through machine learning.<br/>
-feature : action type, action result, position, dist to goal, angle to goal, time등 활용<br/>
-label : score_label=1 if you score within 10 action after each action, and concede_label=1 if you concede within 10 action <br/><br/>

-기존에는 boosting, random-forest, logistic등의 Machine learning기법을 활용하여 득점 및 실점 확률을 예측했지만<br/>
-본 연구에서는 deep-learning기법을 활용하여 예측함으로써 Machine learning기법보다 성능 향상을 검증함<br/><br/>

1.soccer data load.ipynb : StatsBomb데이터를 loading하는 작업<br/>
-본 연구에서는 LaLiga데이터를 활용할 예정<br/>
-game_id = 3,773,689는 특정 플레이 이후 score_label=1, concede_label=1인 데이터가 존재함. 이는 부정확한 데이터이므로 해당 경기 데이터는 제거함<br/><br/>
