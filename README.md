# VAEP(Valuing Actions by Estimating Probabilities)

<Introduction>
-An indicator that uses data called StatsBomb to evaluate the value of all actions : VAEP<br/>
-VAEP is calculated as the difference between the change in scoring probability and the change in conceding probability after predicting the score probability and the concede probability of each action through machine learning.<br/>
-feature : action type, action result, position, dist to goal, angle to goal, time, etc<br/>
-label : score_label=1 if you score within 10 action after each action, and concede_label=1 if you concede within 10 action <br/><br/>

-Previously, machine learning techniques such as boosting, random-forest, and logistic were used to predict the score probability and the concede probability of each action<br/>
-In this study, the performance improvement compared to the Machine learning technique was verified by predicting the scoring probability and the concede probability using the deep-learning technique<br/><br/>

<function><br/>
  
1. soccer data load.ipynb : StatsBomb data loading<br/>
i) In this study, LaLiga data will be used<br/>
ii) game_id = 3,773,689 has data of score_label=1, concede_label=1. This is incorrect data and will remove the game data<br/><br/>

2. soccer computer.ipynb : Define SPADL, Feature, and Label for Train, Valid, and Test data<br/>
i) train : 2004/2005 ~ 2018/2019 season<br/>
ii) valid : 2019/2020 season<br/>
iii) test : 2020/2021 season<br/><br/>

3. soccer data preprocess.ipynb : soccer data preprocessing<br/>
i) Error data preproceing<br/>
ii) Create additional features<br/>
iii) Create labels for multi-classification<br/><br/>

-Modeling & Analysis<br/>
1. ML_BinaryClassification : Machine learning to perform binary-classification<br/>
i) Using the same dataset, CatBoost is used to create a scoring probability model and a concede probability model, respectively<br/>
ii) Calculate the VAEP using the probability of scoring and the conceding<br/>
iii) Quantitative & Qualitative Indicators<br/>
-Qualitative indicators will evaluate the play followed by Valverde's actions, Vasquez's cross, and finally Benzema's shot, starting with Tony Cross's pass at 18:36 of the link below.<br/>
link : https://www.youtube.com/watch?v=EhodpjwTtag&t=1986s<br/><br/>

2. DL_Classification<br/>
i) Deeplearning creates binary-classifications used in previous study and multi-classification proposed in this study<br/>
ii) Calculate the VAEP using the probability of scoring and the conceding<br/>
iii) Quantitative & Qualitative Indicators<br/>
iv) In this study, torchfm/model/Upgrade_Dcn.py & torchfm/layer.py was used. There are many other models and embedding techniques, so please refer to them<br/>
-https://github.com/kitayama1234/Pytorch-Entity-Embedding-DNN-Regressor/blob/master/model.py<br/>
-https://github.com/rixwew/pytorch-fm/tree/master/torchfm<br/><br/>
