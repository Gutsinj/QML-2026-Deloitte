library(TSA)
library(forecast)

setwd("/Users/jacobgutsin/Documents/Deloitte QML Competition")
df <- read.csv("lat_lon_timeseries_2.csv")

# back-transform
df$prediction <- exp(df$prediction) - 1

# extract series
x <- df$prediction

# make it a monthly ts (set these correctly)
x_ts <- ts(x, start = c(2018, 1), frequency = 12)

# quick look
plot(x_ts, type = "l", main = "prediction (monthly)", ylab = "prediction", xlab = "")
acf(x_ts, main = "ACF")
pacf(x_ts, main = "PACF")

# ---- find SARIMA params (non-seasonal + seasonal) ----
# uses forecast::auto.arima (fits SARIMA via Arima), NOT sarimax
fit <- auto.arima(
    x_ts,
    seasonal = TRUE,
    stepwise = FALSE, approximation = FALSE, # more thorough search
    trace = TRUE
)

fit
summary(fit)

# diagnostics
checkresiduals(fit)

# optional: forecast
fc <- forecast(fit, h = 12)
plot(fc)
plot(pacf(x_ts))
