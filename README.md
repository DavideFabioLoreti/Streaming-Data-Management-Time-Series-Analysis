# Streaming-Data-Management-Time-Series-Analysis


## Project Overview  
This project addresses the task of forecasting **hourly pedestrian counts** collected from Australian pedestrian sensors.  
The aim is to predict the final **1,439 hours** (≈ 60 days) of the time series using ARIMA, UCM and Machine Learing models.

---

##  Dataset Description  
The dataset consists of hourly observations with the following structure:

| Column | Description |
|--------|-------------|
| `time` | Timestamp in UTC (YYYY-MM-DD HH:MM:SS) |
| `value` | Number of pedestrians counted during the hour |

- Total observations: **42,767 hours**
- Period: **15 April 2015 → 29 February 2020**
- Key characteristics:
  - strong **daily** seasonality (peak ~16:00)
  - clear **weekly** pattern (minimum on Saturday)
  - **annual** variations (touristic periods)
  - non-stationary variance

---

##  Models Used

#### **ARIMA**
- Implemented in **arima.R**

#### **UCM — Unobserved Components Model**
- Implemented in **UCM.R**

### ** Machine Learning Models **
Several ML approaches were explored:

- **LSTM neural network** (TensorFlow)  
- **KNN Regressor**  
- **Random Forest Regressor**  
- **XGBoost Regressor**


Implemented in **ML.py**.

---

### **3. Final Output**
The final chosen model is the **LSTM**, and the corresponding predictions for the 1,439 future hours are stored in 865309_YYMMDD.csv

---

##  Repository Structure
project-root/
├── arima.R           # Script per la modellazione ARIMA (R)
├── UCM.R             # Script per il modello a componenti non osservate (R)
├── ML.py             # Script per forecasting tramite machine learning (Python)
├── 865309_YYMMDD.csv # File con le previsioni finali
└── README.md         

##  Author  
**Davide Fabio Loreti** — Matricola **865309**  
Master’s Degree in **Data Science**  
Course: *Streaming Data Management and Time Series Analysis*  
University of Milano-Bicocca  
