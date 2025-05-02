# A Reinforcement Learning Application for Statistical Arbitrage
# ATRGStrategies

# -----------------------------------------------------------------
# import dependencies (files and libraries)
rm(list = ls())
#install.packages("plyr")
if (!'pacman' %in% installed.packages()) install.packages('pacman')
library(pacman)

p_load(
  gsubfn,
  plyr,
  purrr,
  purrrlyr,
  R6,
  gridExtra,
  glue,
  discretization,
  assertthat,
  dplyr,
  ggplot2,
  scales,
  colorspace,
  urca
)

# -----------------------------------------------------------------
#region Dependencies

#setwd("C:/Users/user/Desktop/ATRG/ROO/atrg-strategies/src/main/R")
#setwd("C:/Users/Krzysztof/Desktop/ab/ATRGStrategies/atrg-strategies/src/main/R")
data_path <- file.path(getwd(), 'data', '1min')
#data_path <- c("C:\\Users\\user\\Desktop\\ATRG\\data\\data\\")
# biger datahas to be kept outside project path
#
source("./src/rl_agent/object.R")
source("./src/rl_agent/utils.R")
source("./src/rl_agent/getdata.R")
source("./src/rl_agent/cointegration.R")
source("./src/rl_agent/attributes.R")
source("./src/rl_agent/financial_functions.R")
source("./src/rl_agent/discretization.R")
source("./src/rl_agent/state_space.R")
source("./src/rl_agent/reward.R")
source("./src/rl_agent/MCC.R")
source("./src/rl_agent/q_learning.R")
source("./src/rl_agent/tdva.R")
source("./src/rl_agent/rl_utils.R")
