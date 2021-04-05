# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity course for "Machine Learning Engineer with Microsoft Azure".
In this project, we build and optimize an Azure ML pipeline by using the Python SDK and a provided Scikit-learn model. 
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about individuals applying for bank loans. 
The task we set out to accomplish here is to develop a model that, based on the information provided about each individual, predicts whether they will subscribe to a service.

The accuracy in this test was by HyperDrive 91,1% and in the Azure AutoML 91,5% (XGBoostClassifier). 
This seems to be a small difference, but with the last test you get a higher accuracy if you let it calculate the data for a longer time.
So can you say, the XGBoostClassifier is the best in this situation.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
