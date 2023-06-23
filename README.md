# VAEP(Valuing Actions by Estimating Probabilities)

<Introduction>
-An indicator that uses data called StatsBomb to evaluate the value of all actions : VAEP<br/>
-VAEP is calculated as the difference between the change in scoring probability and the change in conceding probability after predicting the score probability and the concede probability of each action through machine learning.<br/>
-feature : action type, action result, position, dist to goal, angle to goal, time등 활용<br/>
-label : score_label=1 if you score within 10 action after each action, and concede_label=1 if you concede within 10 action <br/><br/>

-Previously, machine learning techniques such as boosting, random-forest, and logistic were used to predict the score probability and the concede probability of each action<br/>
-In this study, the performance improvement compared to the Machine learning technique was verified by predicting the scoring probability and the concede probability using the deep-learning technique<br/><br/>

1.soccer data load.ipynb : StatsBomb data loading<br/>
-In this study, LaLiga data will be used<br/>
-game_id = 3,773,689 has data of score_label=1, concede_label=1. This is incorrect data and will remove the competition data<br/><br/>
