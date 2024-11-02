import numpy as np
from quantreg import quantreg

d = np.loadtxt('GEFCOM.txt')
data = d[:, 2]
data_L = d[:, 4]

#Task 1
# Obtain point forecasts from the following naive model:
# ˆ Pd,h = (Pd−1,h + Pd−7,h)/2
# Next calculate forecasts of 5%, 25%, 75% and 95% quantiles
# using the quantile regression model:
# ˆqα
# Pd,h = βα
# 0,h + βα
# 1,h
# ˆ Pd,h.
# Calibrate model parameters for each hour separately, using a
# 364-day calibration window. Calculate the coverage of 50% and
# 90% prediction intervals and the APS of obtained forecasts.

print("TASK 1")
pf_naive1 = np.roll(data, 1*24)
pf_naive2 = np.roll(data, 7*24)

forecast_list = (pf_naive1+pf_naive2)/2
real_list = data

coverages = []
coverages_2 = []

start_day = 0
end_day = 364

for h in range(24):
    X1 = forecast_list[h::24]
    X0 = np.ones(np.shape(X1))
    X = np.column_stack([X0, X1])
    Y = real_list[h::24] #real list jest już od T więc tu indeksy są okej
    Y_cal = Y[end_day:end_day+364]
    #print(len(Y_cal))
    X_cal = X[start_day:end_day]
    #print(len(X_cal))
    Y_fut = Y[end_day:]
    X_fut = X[end_day:]
    taus = [0.05, 0.25, 0.75, 0.95]
    forecasts = np.ones((len(X1)-end_day, len(taus)))

    count = 0
    for t in taus:
        betas = quantreg(X_cal, Y_cal, t)
        for i in range(np.shape(forecasts)[0]):
            forecasts[i ,count] = np.dot(betas, X_fut[i])
        count+=1

    hits = []
    for i in range(np.shape(forecasts)[0]):
        if Y_fut[i] < forecasts[i, 3] and Y_fut[i] > forecasts[i, 0]:
            hits.append(1)
        else:
            hits.append(0)
    coverages.append(np.mean(hits))

    hits_2 = []
    for i in range(np.shape(forecasts)[1]):
        if Y_fut[i] < forecasts[i, 2] and Y_fut[i] > forecasts[i, 1]:
            hits_2.append(1)
        else:
            hits_2.append(0)
    coverages_2.append(np.mean(hits_2))

#print(len(coverages))
print("Coverage for 50% prediction interval:", np.mean(coverages_2).round(5))
print("Coverage for 90% prediction interval:", np.mean(coverages).round(5))


#APS - average pinball score

PS_05 = []
for hour in range(len(forecasts[0])):
    if Y_fut[hour] < forecasts[hour, 0]:
        PS_05.append((0.05-1)*(Y_fut[hour]-forecasts[hour, 0]))
    else:
        PS_05.append(0.05*(Y_fut[hour]-forecasts[hour, 0]))

PS_05_avg = sum(PS_05) / len(PS_05)
print("Avg. PS for q=0.05:", PS_05_avg)

PS_25 = []
for hour in range(len(forecasts[1])):
    if Y_fut[hour] < forecasts[hour, 1]:
        PS_25.append((0.25-1)*(Y_fut[hour]-forecasts[hour, 1]))
    else:
        PS_25.append(0.25*(Y_fut[hour]-forecasts[hour, 1]))

PS_25_avg = sum(PS_25) / len(PS_25)
print("Avg. PS for q=0.25:", PS_25_avg)

PS_75 = []
for hour in range(len(forecasts[2])):
    if Y_fut[hour] < forecasts[hour, 2]:
        PS_75.append((0.75-1)*(Y_fut[hour]-forecasts[hour, 2]))
    else:
        PS_75.append(0.75*(Y_fut[hour]-forecasts[hour, 2]))

PS_75_avg = sum(PS_75) / len(PS_75)
print("Avg. PS for q=0.75:", PS_75_avg)

PS_95 = []
for hour in range(len(forecasts[3])):
    if Y_fut[hour] < forecasts[hour, 3]:
        PS_95.append((0.95-1)*(Y_fut[hour]-forecasts[hour, 3]))
    else:
        PS_95.append(0.95*(Y_fut[hour]-forecasts[hour, 3]))

PS_95_avg = sum(PS_95) / len(PS_95)
print("Avg. PS for q=0.95:", PS_95_avg)

#Task 2
# Obtain forecasts for a 364-day fixed calibration window using the
# following regression model:
# ˆ Pd,h = β0 + β1Pd−1,h + β2Pd−7,h + β3Ld,h
# Next calculate forecasts of 5% and 95% quantiles using the
# quantile regression model:
# ˆqα
# Pd,h = βα
# 0,h + βα
# 1,h
# ˆ Pd,h.
# Calibrate model parameters for each hour separately, using a
# 182-day calibration window. Calculate the APS of obtained
# forecasts.

# HOURLY!

print("")
print("TASK 2")
T = 364 # 364-day fixed calibration window
forecast_list = []
real_list = []
err_h = []

# Loop through each hour
for h in range(24):
    p_hour = data[h::24]
    p_hour_L = data_L[h::24]

    # Calibrate betas using a fixed window of T days for each hour
    cal_data = p_hour[:T]
    cal_data_L = p_hour_L[:T]
    X2 = cal_data[0:T - 7]
    X1 = cal_data[6:T - 1]
    X0 = np.ones(np.shape(X1))
    Y = cal_data[7:T]
    X3 = cal_data_L[7:T]

    X = np.column_stack([X0, X1, X2, X3])
    betas = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    # Forecast for each hour beyond the calibration window
    for day in range(T, len(p_hour)):
        real = p_hour[day]
        X_fut = np.array([1, p_hour[day - 1], p_hour[day - 7], p_hour_L[day]])
        forecast = np.dot(X_fut, betas)
        forecast_list.append(forecast)
        real_list.append(real)
        err_h.append(np.abs(forecast - real))

start_day = 0 #Tutaj start_day to 1 - forecasty punktowe mamy już zrobione więc bierzemy je od początku a nie od 365 (bo to już jest w pętli że ot T=365)
end_day = start_day+182 #calibration window

coverages = []
for h in range(24):
    X1 = forecast_list[h::24]
    X0 = np.ones(np.shape(X1))
    X = np.column_stack([X0, X1])
    Y = real_list[h::24] #real list jest już od T więc tu indeksy są okej
    Y_cal = Y[start_day:end_day]
    X_cal = X[start_day:end_day]
    Y_fut = Y[end_day:]
    X_fut = X[end_day:]
    taus = [0.05, 0.25, 0.75, 0.95]
    forecasts = np.ones((len(X1)-end_day, len(taus)))

    count = 0
    for t in taus:
        betas = quantreg(X_cal, Y_cal, t)
        for i in range(np.shape(forecasts)[0]):
            forecasts[i ,count] = np.dot(betas, X_fut[i])
        count+=1

    hits = []
    for i in range(np.shape(forecasts)[0]):
        if Y_fut[i] < forecasts[i, 3] and Y_fut[i] > forecasts[i, 0]:
            hits.append(1)
        else:
            hits.append(0)
    coverages.append(np.mean(hits))

#print(len(coverages))
print("Coverage for 90% prediction interval:", np.mean(coverages).round(5))

#APS - average pinball score

PS_05_2 = []
for hour in range(len(forecasts[0])):
    if Y_fut[hour] < forecasts[hour, 0]:
        PS_05_2.append((0.05-1)*(Y_fut[hour]-forecasts[hour, 0]))
    else:
        PS_05_2.append(0.05*(Y_fut[hour]-forecasts[hour, 0]))

PS_05_avg_2 = sum(PS_05_2) / len(PS_05_2)
print("Avg. PS for q=0.05:", PS_05_avg_2)

PS_95_2 = []
for hour in range(len(forecasts[3])):
    if Y_fut[hour] < forecasts[hour, 3]:
        PS_95_2.append((0.95-1)*(Y_fut[hour]-forecasts[hour, 3]))
    else:
        PS_95_2.append(0.95*(Y_fut[hour]-forecasts[hour, 3]))

PS_95_avg_2 = sum(PS_95_2) / len(PS_95_2)
print("Avg. PS for q=0.95:", PS_95_avg_2)