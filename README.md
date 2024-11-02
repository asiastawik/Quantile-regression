# Quantile Regression for Price Forecasting

This project implements quantile regression techniques using the GEFCOM dataset to forecast electricity prices and assess the uncertainty associated with these forecasts. The primary focus is on calculating different quantiles to provide a comprehensive view of potential outcomes.

## Project Overview

The project consists of the following key tasks:

1. **Naive Model Point Forecasting**:
   - Obtain point forecasts using the naive model defined as:
     \[
     \hat{P}_{d,h} = \frac{P_{d-1,h} + P_{d-7,h}}{2}
     \]
   - Utilize the quantile regression model to calculate forecasts for the 5%, 25%, 75%, and 95% quantiles:
     \[
     \hat{q}_{\alpha} P_{d,h} = \beta_{0,\alpha,h} + \beta_{1,\alpha,h} \hat{P}_{d,h}
     \]
   - Calibrate the model parameters for each hour separately using a 364-day calibration window and calculate the coverage for 50% and 90% prediction intervals, along with the Average Prediction Score (APS) of the forecasts.

2. **Regression Model Forecasting**:
   - Implement a regression model for a 364-day fixed calibration window.
   - Calculate the 5% and 95% quantiles using the quantile regression model.
   - Calibrate model parameters for each hour separately using a 182-day calibration window and compute the APS of the forecasts.

## Data

The project utilizes the GEFCOM dataset, which includes the following columns: YYYYMMDD, HH, zonal price, system load, zonal load, and day-of-the-week. This dataset is crucial for executing the forecasting tasks and analyzing the results.

## Results

The outputs of the project will include point forecasts, quantile estimates, coverage calculations for prediction intervals, and APS metrics, which will help in understanding the forecasting performance and uncertainty.
