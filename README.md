# ML-Projects
Assignment 1- Polynomial Regression
Assignment 2- SVM
Assignment 3- Neural Networks & CNN on MNIST dataset

1.
Mainly because have gone o very high values since covid19
Before moving further, let me just give a brief idea about different types of mortgage rates.
Interest rate charged from the borrower on mortgage loans by banks
Yield on the agency issued MBS
Difference between the two, basically the profit margins of the banks.
Lets move to the modeling of these mortgage rates.
2. Same 1st line
Basically a 2 step process
Projected as a linear combination of swap rates and
Spread is projected using a mean reversion model where the parameters here have already been fixed based on some calibration done before, here target represent …where its gonna converge based on some half-life. (period say it conclusively)
Moving on to the objectives
3. There were 4 broad targets in this project
Backtest the current models…and report their goodness of fit
Understand….., quality and fitness of current spread model in recent time
Recal……of the current functional form of the spreads to see whether the existing values are still valid or not
And lastly think and implement some alternatives for projecting spreads which could overcome the limitations of the current approach and provide better performance
Coming to the backtesting now
4. We have taken 2 scenarios here, 1st before covid and 2nd after covid
we can see spread is rising abnormally high after covid, and its effect is visible in terms of rmse as well. The model is continuously underpredicting and one more thing to look at here is, since it’s a mean reversion model which constant target, it approaches a fixed value after some time no matter what the market trends are. So it’s a major limitation of this model. So next, we tried to recalibrate its parameters to see any improvement in the performance. 
5. for rec…we have foll a 2 step process..where we have chosen those values of ….which provide lowest rmse on the training set
And then we tested the model with new parameters on the same window as we did before
Now, let’s compare the results after recalibration

6. We can see that, the major change has been in the half life which basically indicates that the projections would converge much faster to the target than before, also, the rmse has improved here but still the model lacks in capturing any volatility in long term, so we will now look at some alternatives which could overcome this flaw in the current approach.
7. so, have considered 3 types of model here,
First a linear model because spreads are correlated with secondary rates
Secondly a mean reversion model because of the mean reverting natue of the spreads
And lastly some non-linear models as well to capture the volatility when spreads were very high
I’ll now describe the models in detail
8. Let me just give a brief idea about each model
0 is the original model
1 is linear model between spread and sec rate
2 is an autoregressive model
3 is a different type of mean reversion models
4 is a power law model
And 5 is model between spread and log transf. of sec rate
Let’s now look into the performance of these models on several training and testing windows
9,10. Here, 2 criterias are used for measuring performance, R2 on training data and RMSE on testing data. Just to make this table easier to comprehend, here are two heat maps for the RMSE and R2 respectively. Here the green cell indicates the best performance and the red cell indicates the worst. In table one, we can that the model 5 is consistently performing very good in terms of rmse , and model 2 is perfoming good in terms of r2, but the model 4 is lacking in both. One more thing to look at here is that performance of model 5 is less sensitive to the training and testing period as opposed to all the other models, hence it’s robust in that sense. And lastly, if you see the performance of model 0, it improves as the testsing period decreases in length which testimonies the fact that the current model is not suited for long term forecasts.
Coming on to the conclusion from this table
11. Finally, we narrowed down model 2 and model 5 based on the analysis, that we just did. We also checked for stationarity and cointegration tests in these regression and found that even though stationarity was lacking in the secondary rates, cointegration was still seen between the spread and the regressors in both the models. Hence the regression made sense.
Just to get an intuition visually, we have some scatter plot to see how the nature of the data is getting affected due to the log transformation here
12, 13. the data here is taken from 2000 to 2020
1st plot is scatter p between….and 2nd plot is the scatter p…between
In the 1st plot the left side indicates the current time regime that we are in..here we can see that there is some curvature as we move left but if you look the 2nd plot the curve has sort of flattened and data has become more linear. This is also indicted in the prediction lines here. The green line here is trying to capture more data which is not the case for the blue line. Hence finding the log transformation was a major breakthrough moment in this project.
Lets now look at the forecast of the new models and the original model during current time.
14
Description of the line color
The plot here shows some interesting results, the model 5 and model 2 are trying to rise in the same manner as the original spread but the original model is revolving around its mean only. Hence the new models are successful in some sense to reflect the market volatility which was missing in the current approach. SO, we believe these new models could considered as suitable alternatives for projecting spreads.
15. Now I’ll just summarize everything that we have done far,
We first back-tested the original models and saw how it deteriorated when the effect of covid19 kicks in due to rising spreads.
Then performed recalibration and saw the parameters changed and the rmse also improved 
Then understood the limitation of the current approach due to its mean reverting behaving and constant underpediction
Finally we tried bunch of alternate models and came with 2 models which showed improved performance in terms of rmse and r-sw and were also robust to the training period.
Hence we covered most of the targets of this project.
16. Finally we also propose few things that could be done further in this project.
We saw the predictions of the current secondary rate projection model and realized that it was underperforming during covid19.
Since we only use swap rates to predict them, treasury rates should also be considered for their projections which might take care of the volatility due stress periods.
And lastly we could analyze the impact of the prepayment modeling in pricing mbs and risk metrics.
Thanks you

