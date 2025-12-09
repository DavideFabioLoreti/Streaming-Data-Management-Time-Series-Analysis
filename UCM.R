library(tidyverse)
library(lubridate)
library(KFAS)
library(readr)

cat("==========================================\n")
cat("UCM MODEL")
cat("==========================================\n\n")


df <- read_csv("C:\\Users\\HP\\Desktop\\università\\DATA SCIENCE\\SECOND YEAR\\STREAMING DATA MANAGEMENT AND TIME SERIES ANALYSIS\\PROGETTO\\student_dataset.csv")


df <- df %>%
  mutate(time = ymd_hms(time, tz = "UTC")) %>%
  arrange(time)

df_obs <- df %>% filter(!is.na(value))
df_future <- df %>% filter(is.na(value))

h <- nrow(df_future)

# 4 HOLIDAY DUMMIES
create_holiday_dummies <- function(dates) {
  tibble(
    time = dates,
    christmas = as.numeric(month(dates) == 12 & day(dates) %in% c(24, 25, 26)),
    new_year = as.numeric((month(dates) == 12 & day(dates) == 31) |
                            (month(dates) == 1 & day(dates) %in% c(1, 2))),
    australia_day = as.numeric(month(dates) == 1 & day(dates) == 26),
    weekend = as.numeric(wday(dates) %in% c(1, 7))
  )
}

holidays_obs <- create_holiday_dummies(df_obs$time)
holidays_future <- create_holiday_dummies(df_future$time)

df_obs <- df_obs %>% left_join(holidays_obs, by = "time")
df_future <- df_future %>% left_join(holidays_future, by = "time")

# FOURIER REGRESSORS
n_obs <- nrow(df_obs)
n_total <- n_obs + h

# Time indices
trainidx <- 1:n_obs
testidx <- (n_obs + 1):n_total
tot <- 1:n_total

# Daily seasonality 
freq_daily <- outer(tot, 1:12) * 2 * pi / 24
cs_daily <- cos(freq_daily[trainidx, ])
si_daily <- sin(freq_daily[trainidx, ])
colnames(cs_daily) <- paste0("cos_d", 1:12)
colnames(si_daily) <- paste0("sin_d", 1:12)

cs_daily_fut <- cos(freq_daily[testidx, ])
si_daily_fut <- sin(freq_daily[testidx, ])
colnames(cs_daily_fut) <- paste0("cos_d", 1:12)
colnames(si_daily_fut) <- paste0("sin_d", 1:12)


freq_weekly <- outer(tot, 1:6) * 2 * pi / (24 * 7)
cs_weekly <- cos(freq_weekly[trainidx, ])
si_weekly <- sin(freq_weekly[trainidx, ])
colnames(cs_weekly) <- paste0("cos_w", 1:6)
colnames(si_weekly) <- paste0("sin_w", 1:6)

cs_weekly_fut <- cos(freq_weekly[testidx, ])
si_weekly_fut <- sin(freq_weekly[testidx, ])
colnames(cs_weekly_fut) <- paste0("cos_w", 1:6)
colnames(si_weekly_fut) <- paste0("sin_w", 1:6)

freq_annual <- outer(tot, 1:1) * 2 * pi / (24 * 365.25)
cs_annual <- cos(freq_annual[trainidx, , drop = FALSE])
si_annual <- sin(freq_annual[trainidx, , drop = FALSE])
colnames(cs_annual) <- paste0("cos_a", 1:1)
colnames(si_annual) <- paste0("sin_a", 1:1)

cs_annual_fut <- cos(freq_annual[testidx, , drop = FALSE])
si_annual_fut <- sin(freq_annual[testidx, , drop = FALSE])
colnames(cs_annual_fut) <- paste0("cos_a", 1:1)
colnames(si_annual_fut) <- paste0("sin_a", 1:1)

# Holiday dummies
holidays_mat <- df_obs %>%
  select(christmas, new_year, australia_day, weekend) %>%
  as.matrix()

holidays_fut <- df_future %>%
  select(christmas, new_year, australia_day, weekend) %>%
  as.matrix()


xreg_train <- cbind(cs_daily, si_daily,
                    cs_weekly, si_weekly,
                    cs_annual, si_annual,
                    holidays_mat)

xreg_full <- rbind(
  xreg_train,
  cbind(cs_daily_fut, si_daily_fut,
        cs_weekly_fut, si_weekly_fut,
        cs_annual_fut, si_annual_fut,
        holidays_fut)
)

cat("Regressors created:\n")
cat(" - Daily Fourier: 24 (12 cos + 12 sin)\n")
cat(" - Weekly Fourier: 12 (6 cos + 6 sin)\n")
cat(" - Annual Fourier: 2 (1 cos + 1 sin)\n")
cat(" - Holiday dummies: 4\n")
cat(" - Total: ", ncol(xreg_train), " regressors\n\n")

cat(" LOG TRANSFORMATION \n")
ytrain <- log(df_obs$value + 1)
vary <- var(ytrain)

cat("==== PRELIMINARY LINEAR MODEL ====\n")
cat("Estimating initial coefficients \n")

lm_init <- lm(ytrain ~ xreg_train)
coef_init <- coef(lm_init)[-1] 
var_coef_init <- vcov(lm_init)[-1, -1] 

cat("Linear model R²:", round(summary(lm_init)$r.squared, 4), "\n")
cat("Coefficients estimated:", length(coef_init), "\n\n")

cat("==== BUILDING UCM MODEL ====\n")
cat("Components:\n")
cat(" - Local level trend (DETERMINISTICO: Q=0)\n")
cat(" - Hourly seasonality (24 hours, dummy coding)\n")
cat(" - Fourier regressors (daily, weekly, annual - K=1)\n")
cat(" - Holiday effects\n\n")

mod_ucm <- SSModel(
  ytrain ~ xreg_train +
    SSMtrend(degree = 1, Q = 0) +
    SSMseasonal(period = 24, sea.type = "dummy", Q = NA),
  H = NA
)

cat("Model created. Checking structure...\n")
cat("State dimension:", dim(mod_ucm$a1)[1], "\n")
cat("Number of observations:", length(ytrain), "\n")
cat("Number of regressors:", ncol(xreg_train), "\n\n")

n_regressors <- ncol(xreg_train)
mod_ucm$a1[1:n_regressors] <- coef_init
mod_ucm$a1[n_regressors + 1] <- ytrain[1] 

mod_ucm$P1inf <- matrix(0, dim(mod_ucm$a1)[1], dim(mod_ucm$a1)[1])
diag(mod_ucm$P1[1:n_regressors, 1:n_regressors]) <- diag(var_coef_init)
diag(mod_ucm$P1[(n_regressors + 1):dim(mod_ucm$a1)[1],
                (n_regressors + 1):dim(mod_ucm$a1)[1]]) <- vary

cat("State vector initialized with linear model estimates\n\n")


fit_ucm <- fitSSM(mod_ucm, inits = rep(log(vary * 0.1), 4),
                  method = "BFGS",
                  control = list(maxit = 1000, trace = TRUE, REPORT = 10))



cat("Convergence:", fit_ucm$optim.out$convergence, "\n")
cat("Log-likelihood:", fit_ucm$optim.out$value, "\n\n")

smo_train <- KFS(fit_ucm$model, smoothing = c("state", "signal"))


level_train <- smo_train$alphahat[, "level"]
fitted_train_log <- smo_train$muhat[, 1]
fitted_train <- exp(fitted_train_log) - 1

residuals_train <- df_obs$value - fitted_train
mae_train <- mean(abs(residuals_train))
rmse_train <- sqrt(mean(residuals_train^2))
mape_train <- mean(abs(residuals_train / df_obs$value[df_obs$value != 0])) * 100 # MAPE gestito per evitare divisione per zero

cat("==== IN-SAMPLE PERFORMANCE (SCALA ORIGINALE) ====\n")
cat("MAE: ", round(mae_train, 2), "\n")
cat("RMSE: ", round(rmse_train, 2), "\n")
cat("MAPE: ", round(mape_train, 2), "%\n\n")

cat("==== GENERATING FORECASTS ====\n")

y_extended <- c(ytrain, rep(NA, h))

mod_forecast <- SSModel(
  y_extended ~ xreg_full +
    SSMtrend(degree = 1, Q = fit_ucm$model$Q[1, 1, 1]) +
    SSMseasonal(period = 24, sea.type = "dummy",
                Q = fit_ucm$model$Q[2, 2, 1]),
  H = fit_ucm$model$H[1, 1, 1]
)

mod_forecast$a1 <- fit_ucm$model$a1
mod_forecast$P1 <- fit_ucm$model$P1
mod_forecast$P1inf <- fit_ucm$model$P1inf

# kalman filter/smoother
smo_forecast <- KFS(mod_forecast, smoothing = c("state", "signal"))

forecasts_log <- smo_forecast$muhat[testidx, 1]

forecasts <- exp(forecasts_log) - 1
forecasts <- pmax(forecasts, 0)

forecast_results <- tibble(
  time = df_future$time,
  value = forecasts
)

cat("Forecast completed for", h, "hours!\n\n")

cat("==== FORECAST STATISTICS ====\n")
print(summary(forecast_results$value))

cat("\nTop hours by mean forecast:\n")
forecast_results %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(mean_forecast = mean(value), .groups = "drop") %>%
  arrange(desc(mean_forecast)) %>%
  head(10) %>%
  print()

cat("\n==== GENERATING PLOTS ====\n")

last_30_days <- df_obs %>%
  filter(time >= max(time) - days(30))

p1 <- ggplot() +
  geom_line(data = last_30_days, aes(x = time, y = value),
            color = "black", linewidth = 0.8) +
  geom_line(data = forecast_results, aes(x = time, y = value),
            color = "red", linewidth = 0.8) +
  geom_vline(xintercept = max(df_obs$time),
             linetype = "dashed", color = "blue", linewidth = 1) +
  labs(
    title = "UCM Forecast (Log-Transformed) for Pedestrian Count",
    subtitle = "Black = Observed | Red = Forecast | Blue line = forecast start",
    x = "Time",
    y = "Pedestrian count"
  ) +
  theme_minimal()

print(p1)

decomp_data <- tibble(
  time = df_obs$time,
  observed = df_obs$value, 
  level = level_train,
  fitted = fitted_train
)

p2 <- ggplot(decomp_data %>% filter(time >= max(time) - days(60))) +
  geom_line(aes(x = time, y = observed), color = "black", alpha = 0.5) +
  geom_line(aes(x = time, y = level), color = "blue", linewidth = 1) +
  labs(
    title = "UCM Level Component (Last 60 days) - SCALA LOGARITMICA",
    subtitle = "Shows underlying log-level (Q=0)",
    x = "Time",
    y = "Level (Log Scale)"
  ) +
  theme_minimal()

print(p2)

p3 <- forecast_results %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(mean_value = mean(value), .groups = "drop") %>%
  ggplot(aes(x = hour, y = mean_value)) +
  geom_line(color = "red", linewidth = 1.2) +
  geom_point(color = "red", size = 3) +
  scale_x_continuous(breaks = 0:23) +
  labs(
    title = "Average Daily Pattern - UCM Forecast (Original Scale)",
    x = "Hour of day",
    y = "Average pedestrian count"
  ) +
  theme_minimal()

print(p3)


file_originale <- "C:\\Users\\HP\\Desktop\\università\\DATA SCIENCE\\SECOND YEAR\\STREAMING DATA MANAGEMENT AND TIME SERIES ANALYSIS\\PROGETTO\\865309_YYYMMDD.csv"

df_originale <- read_csv(file_originale)

df_originale <- df_originale %>%
  left_join(forecast_results %>% rename(forecast = value), by = "time")

write_csv(df_originale, file_originale)

cat("\n==========================================\n")
cat("CM MODEL - FINAL SUMMARY\n")
cat("==========================================\n")
cat("Training period:", as.character(min(df_obs$time)), "->",
    as.character(max(df_obs$time)), "\n")
cat("Forecast period:", as.character(min(df_future$time)), "->",
    as.character(max(df_future$time)), "\n")
cat("Hours forecasted:", h, "\n")
cat("Regressors:", ncol(xreg_train), "\n")
cat(" - Daily Fourier (K=12): 24\n")
cat(" - Weekly Fourier (K=6): 12\n")
cat(" - Annual Fourier (K=1): 2\n")
cat(" - Holidays: 4\n")
cat("Components:\n")
cat(" - Local level trend (stazionario, Q=0)\n")
cat(" - Hourly seasonality (24h)\n")
cat("Transformation: Log(y+1)\n")
cat("In-sample MAE:", round(mae_train, 2), "\n")
cat("In-sample RMSE:", round(rmse_train, 2), "\n")
cat("==========================================\n")
cat("✓ UCM MODEL COMPLETED SUCCESSFULLY!\n")
cat("==========================================\n")