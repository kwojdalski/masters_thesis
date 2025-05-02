source('./src/rl_agent/pre_run.R')

#c_currencies <- list.files("C:/Users/user/Desktop/UCL final project/data/1min") # 1min data
c_currencies <- c("EURGBP", "EURUSD", "USDJPY", "EURJPY", "CHFJPY", "GBPUSD", "AUDJPY", "EURCAD")
#c_currencies <- randomObs(c_currencies, 12) # 12 randon currencies
#c_currencies <- c_currencies[! c_currencies %in% c(c_currencies1)]
#df_currencies <- GetQuotes(c_currencies, data_path, ".csv")

# New data 08.2015
#c_currencies <- c("EURGBPask", "EURUSDask", "USDJPYask", "EURJPYask", "CHFJPYask", "GBPUSDask", "AUDJPYask", "EURCADask")
df_currencies <- GetQuotes(c_currencies, data_path, ".csv")

#endregion
# -----------------------------------------------------------------
#region Cointegration extraction

# Get all cointegrated pairs on a specified base interval

train_period <- c(1,1000)
test_period <- c(1001,300)
debugonce(GetCointPairs)

pairs <- GetPairs(df_currencies, c_currencies, train_period) %>%
  purrr::transpose() %>%
  lapply(function(x) invoke(cbind, x) %>% as.data.frame())



#endregion
# -----------------------------------------------------------------
#region Attributes extraction

#Create the data frame of attributes for all cointegrated pairs
i_delay <- 2
l_attributes <- AttributesAll(pairs[[3]], colnames(pairs[[3]]), i_delay)
str(l_attributes)
#endregion
# -----------------------------------------------------------------
#region RL backtest data preparation

# Pick one of the pairs and it's attributes

pair_ind <- 1

length(l_attributes[[3]])
train_pair <- pairs[[1]][, pair_ind]
train_features <- as.data.frame(l_attributes[pair_ind])[train_period[1]:train_period[2], ]

test_pair <- pairs[[2]][, pair_ind]
test_features <- as.data.frame(l_attributes[pair_ind])[test_period[1]:test_period[2],]

plot.ts(test_pair)
plot.ts(train_pair)

#endregion
# -----------------------------------------------------------------
#region MCC test





Nepisodes <- 10
cost <- 0
debugonce(MCCbootstrap)
pretrained_agent <- MCCbootstrap(coint)


pretrained_pos <- pretrained_agent[[2]]
pretrain_return <- Ret(train_pair, coef, tail(pretrained_pos, nrow(train_pair)), 0)
PlotAgentResult(train_pair, train_spread, pretrain_return,'Monte Carlo Control')


tested_agent <- MCCbootstrap(test_spread, test_pair, coef, test_features, Nepisodes, cost, pretrained_agent)
test_pos <- tested_agent[[2]]
test_return <- Ret(test_pair, coef, tail(test_pos, nrow(test_pair)), 0)
PlotAgentResult(test_pair, test_spread, test_return, 'Monte Carlo Control')

# PRZEKAZANIE VSA DO TESTOWEGO OKRESU. Jak to ma jakies parametry typu alpha to to powinniem byc hyper parametr
# jezeli mamy zamiar ogolnie optymalizowac. Czy dlugosc epizodu i mierzenia kointegracji (base i test period) mogla
# by byc wstawiona jako parametry dla agenta?

#endregion
# -----------------------------------------------------------------
#region Qlearning test

Nepisodes <- 5
cost <- 0

#experience replay - nie przejmuje sie co sie dzieje do moment uduzego skoku, a potem juz tylko idzie w gore.

pretrained_agent <- Qcontrol(train_pair, coef, train_features, Nepisodes, cost)
pretrained_pos <- pretrained_agent[[2]]
pretrain_return <- Ret(train_pair, coef, tail(pretrained_pos, nrow(train_pair)), 0)
PlotAgentResult(train_pair, train_spread, pretrain_return,'Qlearning Control')


tested_agent <- Qcontrol(test_pair, coef, test_features, 1, cost, pretrained_agent)
test_pos <- tested_agent[[2]]
test_return <- Ret(test_pair, coef, tail(test_pos, nrow(test_pair)), 0)
PlotAgentResult(test_pair, test_spread, test_return, 'Qlearning Control')
