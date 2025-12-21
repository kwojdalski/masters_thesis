# Data Used in "Can Artificial Intelligence Trade the Stock Market?"

This document outlines the data sources and features utilized in the research paper.

## 1. Asset Classes

The study analyzed data from five distinct financial assets:
*   **Currency Pairs (FOREX):**
    *   EUR/JPY
    *   EUR/USD
    *   USD/JPY
*   **Stock Index:**
    *   S&P 500
*   **Cryptocurrency:**
    *   Bitcoin (BTC)

## 2. Data Granularity and Frequency

*   **Type:** Daily data.
*   **Actions:** Agents executed trades on a daily frame, specifically at midnight US time (GMT -4 hours), every 24 hours.

## 3. Data Period

*   **Study Period:** 2019-2023 for all assets (used for validation and out-of-sample testing).
*   **Data Collection Start Dates:**
    *   **Currencies and S&P 500:** From the year 2005.
    *   **Bitcoin:** From January 1, 2013.

## 4. Input Features for Models

The input features provided to the DRL models varied slightly by asset but generally included:
*   **Current Market Position:** To inform the agent about its existing holdings.
*   **Returns:** Calculated based on closing prices.
*   **OHLC Data:** Open, High, Low, Close prices.
*   **Technical Indicators:**
    *   Relative Strength Index (RSI)
    *   Moving Averages
    *   Average True Range (ATR)
    *   Moving Average Convergence/Divergence (MACD)
*   **Time-Related Trends:** Sine and cosine transformations of time to capture seasonality.

## 5. Data Preprocessing

*   All data inputs were **standardized or normalized**.
*   **Careful measures were taken to prevent look-ahead bias**, although specific methods were not detailed beyond mentioning this general principle.

## 6. Training and Testing Methodology

*   **Optimization:** A **moving forward optimization** (also known as rolling window optimization) approach was used.
*   **Windows:** The data was organized into five sequential windows.
*   **Validation/Test Split:** Each window comprised:
    *   One year of data for validation.
    *   One subsequent year for out-of-sample testing.
*   **Example Progression:**
    *   Validation 2018, Test 2019
    *   Progression through 2020, 2021, 2022, and 2023 for subsequent testing periods.
*   **Agent Reinitialization:** For each new window, the agent was re-trained from scratch to prevent bias or overfitting from previous training phases.
