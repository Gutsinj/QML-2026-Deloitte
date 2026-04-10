library(TSA)

setwd("/Users/ethanabelev/Documents/Deloitte QML/QML-2026-Deloitte")
df <- read.csv("lat_lon_timeseries_2.csv")   # adjust filename/path
str(df)
head(df)
plot(df,type="l")

df <- df["prediction"]
pacf(df)