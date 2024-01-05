# Customer-Churn-Prediction
The Customer Churn Prediction project focuses on developing and evaluating machine learning models to predict customer churn in a subscription-based business model. The dataset includes various customer-related features such as gender, tenure, contract terms, payment methods, and monthly charges.

Logistic Regression:
The Logistic Regression model achieved an accuracy of 76.37%. However, it shows a challenge in predicting positive churn cases (Yes) with a lower recall (15%) compared to non-churn cases (No). The precision for churn cases is reasonable (77%), indicating that when the model predicts churn, it is often correct. The weighted F1-score is 0.70.

Decision Trees:
The Decision Trees model performed slightly better, achieving an accuracy of 77.71%. It exhibits improved recall for churn cases (Yes) at 51%, but there is still room for enhancement. The model shows a balanced performance with higher precision and recall for non-churn cases (No). The weighted F1-score is 0.77.

Neural Networks (MLP):
The Neural Networks (MLP) model achieved an accuracy of 73.53%. However, it struggles to correctly identify churn cases (Yes) with a recall of 0%. This indicates a significant limitation in capturing positive churn instances. The precision for non-churn cases (No) is reasonable. The weighted F1-score is 0.62.

Conclusion:
The models present varying levels of performance, and further optimization is required to enhance the prediction of customer churn. The Decision Trees model exhibits the best overall performance among the three models. Strategies to improve recall for positive churn cases and feature engineering may contribute to refining the models and providing more accurate predictions.






