# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity course for "Machine Learning Engineer with Microsoft Azure".
In this project, we build and optimize an Azure ML pipeline by using the Python SDK and a provided Scikit-learn model. 
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about individuals applying for bank loans. 
The task we set out to accomplish here is to develop a model that, based on the information provided about each individual, predicts whether they will subscribe to a service.

The accuracy in this test was by HyperDrive 91,1% and in the Azure AutoML 91,5% (XGBoostClassifier). 
This seems to be a small difference, but with the last test you get a higher accuracy if you let it calculate the data for a longer time (>40 minutes).
**So can you say, the XGBoostClassifier is the best in this situation.**


## Scikit-learn Pipeline
The Scikit-learn pipeline obtains the provided data in csv format from the provided URL. 
Following data download, a number of data cleaning steps are carried out including:

- Removing NA from the dataset.
- Encoding job titles, contact, and education variables.
- Encoding a number of other categorical variables.
- Encoding months of the year.
- Encoding the target variable.

Once the data has been prepared it is split into a training and test set. 
A test set size of 28% of total entries was selected as a compromise between ensuring adequate representation in the test data and providing sufficient data for model training.

I am specified the parameter sampler in detail:
```
...
ps = RandomParameterSampling({
    "--C" : choice(0.01, 0.1, 1.2,1.5),
    "--max_iter" : choice(20, 40, 60, 100,150,200,250)
})
...
```
```
...
train_data, test_data = train_test_split(full_data,test_size=0.28)
...
```
  


## AutoML
I am have used in the AutoML option the task as "clasification" and primary metric as "accuracy" and iterations as 5 (for a quick test time, better is higher 40), 
the timeout for this final test was by 40 minutes. I think a longer time and more iterations, make better results.


## Pipeline comparison
In our experiments, the performance of 2 options is comparable, with HyperDrive option having an accuracy of 91,1 % while AutoML option having a better accuracy of 91,5 %. Even with consideration to the limitation of our experiments, time and limited data points, we can say that Azure AutoML will result in a better performance as AutoML go through and test multiple classifciation models while the HyperDrive option just uses the Logistic Regression Algorithm. 


## Future work
With HyperDrive, we can increase the max_total_runs parameter allowing us to go through more hyperparameter options. More time better reults, but only testing one classifation model. We can are doing intial search using Random Sampling, we can refine and narrow our search to find the best hyperparameters and use grid sampling to do so. 

With Azure AutoML, we can increase the number of iteration and the test runtime allowing us to go through more models supported by AutoML for classifcation. Trying more models will help us find the best model for this datasample. So can we testing in shorter time more classifcation models then in HyperDrive. 

