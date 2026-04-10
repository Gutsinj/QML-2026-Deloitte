library(TSA)

setwd("/Users/ethanabelev/Documents/Deloitte QML/QML-2026-Deloitte")
df <- read.csv("lat_lon_timeseries_2.csv")   # adjust filename/path
df["prediction"] <- exp(df["prediction"]) - 1
str(df)
head(df)
plot(df,type="l")

df <- df["prediction"]
#acf(df)