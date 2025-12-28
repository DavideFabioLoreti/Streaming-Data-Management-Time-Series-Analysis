library(tidyverse)
library(lubridate)
library(forecast)
library(readr)
# ========================================
# ARIMA WITH FOURIER - FINAL MODEL (Updated)
# ========================================

cat("==========================================\n")
cat("   FOURIER ARIMA \n")
cat("==========================================\n\n")

# LOAD DATASET
# Using file uploaded by user
df <- read_csv("C:\\Users\\HP\\Desktop\\università\\DATA SCIENCE\\SECOND YEAR\\STREAMING DATA MANAGEMENT AND TIME SERIES ANALYSIS\\PROGETTO\\student_dataset.csv")

# CLEANING AND SPLITTING
df <- df %>%
  mutate(time = ymd_hms(time, tz = "UTC")) %>%
  arrange(time)

df_obs    <- df %>% filter(!is.na(value))
df_future <- df %>% filter(is.na(value))

h <- nrow(df_future)

# DATASET INFORMATION
cat("==== DATASET INFO ====\n")
cat("Observed data:\n")
cat("  Period:", as.character(min(df_obs$time)), "->", as.character(max(df_obs$time)), "\n")
cat("  N. observations:", nrow(df_obs), "\n")
cat("  N. days:", round(as.numeric(difftime(max(df_obs$time), min(df_obs$time), units = "days")), 1), "\n\n")

cat("Data to forecast:\n")
cat("  Period:", as.character(min(df_future$time)), "->", as.character(max(df_future$time)), "\n")
cat("  N. observations:", h, "\n")
cat("  N. days:", round(as.numeric(difftime(max(df_future$time), min(df_future$time), units = "days")), 1), "\n\n")

# CREATE DUMMY VARIABLES FOR AUSTRALIAN HOLIDAYS
create_holiday_dummies <- function(dates) {
  tibble(
    time = dates,
    # Christmas and surrounding days (Dec 24–26)
    christmas = as.numeric(month(dates) == 12 & day(dates) %in% c(24, 25, 26)),
    # New Year and adjacent dates (Dec 31 – Jan 2)
    new_year = as.numeric(
      (month(dates) == 12 & day(dates) == 31) |
        (month(dates) == 1 & day(dates) %in% c(1, 2))
    ),
    # Australia Day (Jan 26)
    australia_day = as.numeric(month(dates) == 1 & day(dates) == 26),
    # Weekend
    weekend = as.numeric(wday(dates) %in% c(1, 7))
  )
}

# Apply to series
cat("==== CREATING REGRESSORS ====\n")
holidays_obs <- create_holiday_dummies(df_obs$time)
holidays_future <- create_holiday_dummies(df_future$time)

df_obs <- df_obs %>% left_join(holidays_obs, by = "time")
df_future <- df_future %>% left_join(holidays_future, by = "time")

# Count holidays in observed data
cat("Holidays in observed dataset:\n")
cat("  Christmas:", sum(df_obs$christmas), "hours\n")
cat("  New Year:", sum(df_obs$new_year), "hours\n")
cat("  Australia Day:", sum(df_obs$australia_day), "hours\n")
cat("  Weekend:", sum(df_obs$weekend), "hours\n\n")


# CREATE TIME SERIES (ADDING ANNUAL SEASONALITY)
y_full <- ts(df_obs$value, frequency = 24)
y_full_weekly <- ts(df_obs$value, frequency = 24*7)
# Annual frequency: 24 hours * 365.25 days
y_full_annual <- ts(df_obs$value, frequency = 24*365.25) 

#  FOURIER PARAMETERS 
K_daily  <- 12  # 24 regressors
K_weekly <- 6   # 8 regressors
K_annual <- 1   # 2 regressors

cat("==== UPDATED MODEL PARAMETERS ====\n")
cat("K_daily:", K_daily, "(24 Fourier regressors for daily pattern)\n")
cat("K_weekly:", K_weekly, "(8 Fourier regressors for weekly pattern)\n")
cat("K_annual:", K_annual, "(4 Fourier regressors for yearly pattern) **NEW**\n")
cat("Holidays: 4 dummy variables (christmas, new_year, australia_day, weekend)\n\n")

# BUILD REGRESSOR MATRIX (INCLUDING ANNUAL CYCLE)
# Daily Fourier
xreg_daily <- fourier(y_full, K = K_daily)

# Weekly Fourier
xreg_weekly <- fourier(y_full_weekly, K = K_weekly)

# Annual Fourier 
xreg_annual <- fourier(y_full_annual, K = K_annual)

# Holidays
holidays_full_mat <- df_obs %>% 
  select(christmas, new_year, australia_day, weekend) %>%
  as.matrix()

xreg_full <- cbind(xreg_daily, xreg_weekly, xreg_annual, holidays_full_mat)

cat("Regressor matrix dimensions:", dim(xreg_full), "\n")
cat("  → Total regressors:", ncol(xreg_full), "\n\n")

# TRAIN FINAL ARIMA
cat("==== TRAINING MODEL ====\n")
cat("Fitting ARIMA with extended Fourier regressors...\n")

fit_final <- auto.arima(
  y_full,
  xreg = xreg_full,
  seasonal = FALSE,
  stepwise = TRUE,
  approximation = TRUE,
  trace = TRUE
)

cat("Selected model:", fit_final$method, "\n\n")

# Summary
cat("==== MODEL SUMMARY ====\n")
summary(fit_final)

# Residual diagnostics
cat("\n==== RESIDUAL DIAGNOSTICS (Goal: high p-value) ====\n")
checkresiduals(fit_final)

# In-sample errors
train_mae <- mean(abs(residuals(fit_final)))
train_rmse <- sqrt(mean(residuals(fit_final)^2))
cat("\nIn-sample errors (training):\n")
cat("  MAE:", round(train_mae, 2), "\n")
cat("  RMSE:", round(train_rmse, 2), "\n\n")

# FUTURE REGRESSORS
cat("==== FORECAST GENERATION ====\n")

holidays_future_mat <- df_future %>% 
  select(christmas, new_year, australia_day, weekend) %>%
  as.matrix()

xreg_future_final <- cbind(
  fourier(y_full, K = K_daily, h = h),
  fourier(y_full_weekly, K = K_weekly, h = h),
  fourier(y_full_annual, K = K_annual, h = h), # NEW
  holidays_future_mat
)

cat("Generating forecasts for", h, "future hours...\n")

# FINAL FORECAST
fc_final <- forecast(
  fit_final,
  xreg = xreg_future_final,
  h = h
)

#  RESULTS TABLE
final_predictions <- tibble(
  time = df_future$time,
  value = pmax(as.numeric(fc_final$mean), 0)  # avoid negative values
)

cat("Forecast completed!\n\n")

# FORECAST STATISTICS
cat("==== FORECAST STATISTICS ====\n")
cat("Forecast summary:\n")
print(summary(final_predictions$value))

cat("\nDistribution by hour:\n")
final_predictions %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(
    mean_forecast = mean(value),
    median_forecast = median(value),
    min_forecast = min(value),
    max_forecast = max(value),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_forecast)) %>%
  head(10) %>%
  print()


cat("\n==== HOLIDAYS IN FORECAST ====\n")
forecast_holidays <- final_predictions %>%
  left_join(df_future %>% select(time, christmas, new_year, australia_day), by = "time") %>%
  filter(christmas == 1 | new_year == 1 | australia_day == 1) %>%
  mutate(
    day = date(time),
    holiday = case_when(
      new_year == 1 & day(time) %in% c(1, 2) ~ "New Year",
      australia_day == 1 ~ "Australia Day",
      TRUE ~ "Other"
    )
  ) %>%
  group_by(day, holiday) %>%
  summarise(
    mean_value = mean(value),
    .groups = "drop"
  )

print(forecast_holidays)

# VISUALIZATION
cat("\n==== GENERATING PLOTS ====\n")

# Plot 1: Last 30 days observed + forecast
last_30_days <- df_obs %>%
  filter(time >= max(time) - days(30))

p1 <- ggplot() +
  geom_line(data = last_30_days, aes(x = time, y = value), 
            color = "black", linewidth = 0.8) +
  geom_line(data = final_predictions, aes(x = time, y = value), 
            color = "blue", linewidth = 0.8) +
  geom_vline(xintercept = max(df_obs$time), 
             linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Extended Fourier ARIMA Forecast",
    subtitle = "Black = Observed | Blue = Forecast | Red line = start of forecast",
    x = "Time",
    y = "Pedestrian count"
  ) +
  theme_minimal()

print(p1)

# Plot 2: Average daily pattern
p2 <- final_predictions %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(mean_value = mean(value), .groups = "drop") %>%
  ggplot(aes(x = hour, y = mean_value)) +
  geom_line(color = "blue", linewidth = 1.2) +
  geom_point(color = "blue", size = 3) +
  scale_x_continuous(breaks = 0:23) +
  labs(
    title = "Average Daily Pattern - Forecast",
    x = "Hour of day",
    y = "Average pedestrian count"
  ) +
  theme_minimal()

print(p2)

# Plot 3: Zoom on first 2 forecast weeks
p3 <- final_predictions %>%
  filter(time <= min(time) + days(14)) %>%
  ggplot(aes(x = time, y = value)) +
  geom_line(color = "blue", linewidth = 0.8) +
  labs(
    title = "Zoom: First 2 Weeks of Forecast",
    subtitle = "January 2020",
    x = "Time",
    y = "Pedestrian count"
  ) +
  theme_minimal()

print(p3)

# save results in the local machine
cat("\n==== SAVING RESULTS ====\n")

forecast_results <- final_predictions %>%
  filter(time %in% df_future$time)

output_file <- "C:\\Users\\HP\\Desktop\\università\\DATA SCIENCE\\SECOND YEAR\\STREAMING DATA MANAGEMENT AND TIME SERIES ANALYSIS\\PROGETTO\\865309_YYYMMDD.csv"
write_csv(forecast_results, output_file)
cat("✓ Forecast-only file saved:", output_file, "\n")


# final report
cat("\n==========================================\n")
cat("            FINAL SUMMARY\n")
cat("==========================================\n")
cat("Model:", fit_final$method, "\n")
cat("Training period:", as.character(min(df_obs$time)), "->", as.character(max(df_obs$time)), "\n")
cat("Forecast period:", as.character(min(df_future$time)), "->", as.character(max(df_future$time)), "\n")
cat("Hours forecasted:", h, "\n")
cat("Regressors used:", ncol(xreg_full), "\n")
cat("  - Fourier daily: K =", K_daily, "(Increased)\n")
cat("  - Fourier weekly: K =", K_weekly, "(Increased)\n")
cat("  - Fourier annual: K =", K_annual, "(NEW)\n")
cat("  - Holidays: 4 dummy variables\n")
cat("In-sample MAE:", round(train_mae, 2), "\n")
cat("Output file:", output_file, "\n")
cat("==========================================\n")
cat("✓ PROCESS SUCCESSFULLY COMPLETED!\n")
cat("==========================================\n")

# show preview of forecast
cat("\nFirst 10 predictions:\n")
print(head(final_predictions, 10))

cat("\nLast 10 predictions:\n")
print(tail(final_predictions, 10))

plot(final_predictions, type = "l")

last_30_days <- df_obs %>% filter(time >= max(time) - days(30))

p1 <- ggplot() +
  geom_line(data = last_30_days, aes(x = time, y = value), color = "black", linewidth = 0.8) +
  geom_line(data = final_predictions, aes(x = time, y = value), color = "blue", linewidth = 0.8) +
  geom_vline(xintercept = max(df_obs$time), linetype = "dashed", color = "red", linewidth = 1) +
  labs(title = "Extended Fourier ARIMA Forecast", x = "Time", y = "Pedestrian count") +
  theme_minimal()
print(p1)




library(ggplot2)
library(dplyr)
library(lubridate)

cat("\n==== FORECAST vs JAN/FEB Series ====\n")


df_obs_janfeb <- df_obs %>%
  filter(month(time) %in% c(1, 2)) %>%
  mutate(year = as.character(year(time)),
         type = "Storico")


forecast_janfeb <- final_predictions %>%
  mutate(year = "Forecast",
         type = "Forecast")


df_plot <- bind_rows(df_obs_janfeb, forecast_janfeb)

df_plot <- df_plot %>%
  mutate(reference_time = make_datetime(
    year = 2000,
    month = month(time),
    day = day(time),
    hour = hour(time),
    min = minute(time),
    sec = second(time)
  ))


ggplot() +
  geom_line(data = df_plot %>% filter(type == "Storico"),
            aes(x = reference_time, y = value, group = year, color = year),
            linetype = "dashed", linewidth = 0.8) +
  geom_line(data = df_plot %>% filter(type == "Forecast"),
            aes(x = reference_time, y = value),
            color = "black", linewidth = 1.2) +
  scale_color_manual(values = rainbow(length(unique(df_obs_janfeb$year)))) +
  labs(
    title = "Confronto pattern Gen-Feb: Storico vs Previsione",
    subtitle = "Linee storiche (tratteggiate colorate) vs forecast (linea nera continua)",
    x = "Tempo normalizzato (mese/giorno/ora)",
    y = "Pedestrian Count"
  ) +
  theme_minimal(base_size = 12)







