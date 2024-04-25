# Heart-Disease Prediction Model
The Repository contains a Data science project for predicting heart disease using Machine learning techniques. the project includes Exploratory data analysis (EDA), tuning hyperparameters and evaluation metrics such as ROC and AUC curves.
The data-set used for training and evaluation contains information about the patients `Age`,`Sex`,`Chest Pain type`,`Resting Blood pressure`,`serum Cholestrol level`, `Maximum heart rate achieved` and many other affecting features which might positively and negatively impact the heart of the person.

## Problem Statement
The goal of the project is to use the technologies for betterment to society by impacting detection of heart disease before it actually happens, as the facts states that heart disease is the leading reason for the death which is then followed by cancer and other things so using the previous data of patients and using the right technology (Machine learning) we could try to detect the disease in a very early stage. which is the aim for this project.

## Approach
The project follows a structured machine learing pipeline:
1) `Data Collection`: The dataset used for training and evalution is obtained from Kaggle itself which has about 13 columns in it which positively or negatively impact the target column. thus in total we have 14 columns and total entries is above `1k`. <br>
Link : https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction
2) `Exploratory Data Anlysis (EDA)`: An Exploratory analysis of the dataset is conducted to gain insights into the distribution of variables, correlations and potential patterns.
Here in this section just for sample example we have gained insights in chest pain column and plotting it on interactive graph.
where 0 : Typical angina: Chest pain related decrease blood supply to the heart.
      1 : Atypical angina: Chest pain not related to heart.
      2 : Non-Anginal pain: Typically esophageal spasms (non-heart related).
      3 : Asympotomatic : chest pain not showing signs of disease. 
and then plotting this information in-terms of in which cases for example having which kind of chest pain could lead most likely to have a heart disease.<br>
`Insights`: Where in i discovered that most time when you have type 0 chest pain the chances of that leading actually to a heart disease is comparatively less vs type 1 and 2 states that if you have such kind of pain in chest you could not risk of waiting, advised to concult a professional as soon as possible.

3) `Model Selection`: It is a very important step for this project as not always all the models are worth trying on as it is time consuming and tideous work to showcase, However for this project we have used few more machine learning Algorithm for better understanding of each and every Algorithm as understand how it work and processes. <br>
   We have in total 5 Algorithms:
   1) Logistic Regression :
   2) Random Forest Classifier :
   3) Gaussian Naive Bayes :
   4) XGBoost Classifier :
   5) Decision tree :
4) `Hyperparameter Tuning` : Based on the above models we are then trying to work on the factors/nobes by which we can try to improve the performance of the algorithms.
And for that we are using this technique which is nothing but Tuning/regulating the nobes whihc we can change inorder to get out the best performance from the Algorithm and for that purpose we are experimenting by using 2 techniques:<br>
    1) GridSearchCV
    2) RandomizedSearchCV 
5) `Model Evaluation` : The trained models are then evaluated using various metrics such as accuracy, precision, recall, F1-Score, and Cross-Validation Techniques as employed to ensure robustness of the results.
6) `Feature Importance` : Feature importance is been done to analyze which features from the data are actually contributing most to the model's prediction.
## Results
* The `Logistic Regression` model achieved the highest accuracy of `85%` on the test data.
* `Decision Tree` and `XGBoost Classifier` are the one which gives the best outcomes ie `90% Accuracy` and `91% Accuracy`       respectively
* `Random Forest Classifier` and `Gaussian Naive Bayes` are the Algorithm where we got `87%` and `80%` accuracy respectively.
*  Whereas `KNN` is one where we are getting less accuracy compare to others.

## Conclusion:
Developing an accurate predictive model for heart disease detection requires a systematic approaches involving data preprocessing, model selection and hyperparameter tuning. While the current models show promising results, there is room for improvement through continued experimentation and refinement.

## Contribution
Sujal thakkar : thakkar.su@northeastern.edu, sujalthakkar95@gmail.com
  
