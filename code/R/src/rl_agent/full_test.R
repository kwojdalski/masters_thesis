# A Reinforcement Learning Application for Statistical Arbitrage
# ATRGStrategies

# -----------------------------------------------------------------
# import dependencies (files and libraries)

#install.packages("plyr")
library(plyr) # data manipulations
setwd("C:/Users/user/Desktop/ATRG/ROO/atrg-strategies/src/main/R")
data_path <- paste(getwd(), c("/data/1min/"), sep ="")

source(file = "getdata.R")
source(file = "cointegration.R")
source(file = "attributes.R")
source(file = "financial_functions.R")
source(file = "discretization.R")
source(file = "state_space.R")
source(file = "reward.R")
source(file = "MCC.R")

# -----------------------------------------------------------------
#region Raw data


#c_currencies <- list.files("C:/Users/user/Desktop/UCL final project/data/1min") # 1min data
c_currencies <- c("EURGBP", "EURUSD", "USDJPY", "EURJPY", "CHFJPY", "GBPUSD", "AUDJPY", "EURCAD")
#c_currencies <- randomObs(c_currencies, 12) # 12 randon currencies
#c_currencies <- c_currencies[! c_currencies %in% c(c_currencies1)]
df_currencies <- GetQuotes(c_currencies, data_path, ".csv")


#endregion
# -----------------------------------------------------------------
#region Cointegration extraction

train_length <- 3000
test_length <- 200

Nepisodes <- 2
cost <- 0

i_delay <- 15

all_agent_mean_ret <- c()
all_asset_mean_ret <- c()
previous_agent_ret <- 1
previous_pair_ret <- 1

i <- 1
while((train_length + test_length*i) < nrow(df_currencies)){

  # After each test_length data points reestimate cointegration
  # on a window of length train_length
  train_period <- c(test_length*(i-1) + 1, train_length + test_length*(i-1))
  test_period <- c(train_length + test_length*(i-1) + 1, train_length + test_length*i)

  coint_pairs <- GetCointPairs(df_currencies, c_currencies, train_period, 3)
  coint_spreads <- CointSpreads(coint_pairs, df_currencies, train_period, test_period)

  train_spreads <- as.data.frame(coint_spreads[[1]])
  test_spreads <- as.data.frame(coint_spreads[[2]])
  all_spreads <- as.data.frame(coint_spreads[[3]])
  coefs <- as.data.frame(coint_spreads[[4]])

  l_attributes <- AttributesAll(all_spreads, colnames(all_spreads), i_delay)

  # Backtest each cointegrated pair
  agent_return <- list()
  pair_return <- list()

  for(j in 1:ncol(coint_pairs)){

    print(j / ncol(coint_pairs))

    pair_ind <- j
    coint_pair <- coint_pairs[,pair_ind]
    coef <- coefs[,pair_ind]

    train_pair <- df_currencies[c(train_period[1]:train_period[2]), coint_pair]
    train_spread <- train_spreads[[pair_ind]]
    train_features <- as.data.frame(l_attributes[pair_ind])[c(train_period[1]:train_period[2]),]

    test_pair <- df_currencies[c(test_period[1]:test_period[2]), coint_pair]
    test_spread <- test_spreads[[pair_ind]]
    test_features <- as.data.frame(l_attributes[pair_ind])[c(test_period[1]:test_period[2]),]

    #endregion
    # -----------------------------------------------------------------
    #region MCC test

    pretrained_agent <- MCCbootstrap(train_spread, train_pair, coef, train_features, Nepisodes, cost)
    pretrained_pos <- pretrained_agent[[2]]
    pretrain_return <- Ret(train_pair, coef, train_spread, tail(pretrained_pos, nrow(train_pair)), 0)
    #PlotAgentResult(train_pair, train_spread, pretrain_return)

    tested_agent <- MCCbootstrap(test_spread, test_pair, coef, test_features, Nepisodes, cost, pretrained_agent)
    test_pos <- tested_agent[[2]]
    test_return <- Ret(test_pair, coef, test_spread, tail(test_pos, nrow(test_pair)), 0, previous_ret=previous_agent_ret)
    agent_return[[paste(coint_pairs[1,j],coint_pairs[2,j])]] <- test_return

    #O jeden za duzo rowMeans w tym miesjcu?
    pair_return[[paste(coint_pairs[1,j],coint_pairs[2,j])]] <- rowMeans(apply(test_pair, 2, ret_asset,
                                                                              previous_ret=previous_pair_ret))
  }
  agent_return <- rowMeans(as.data.frame(agent_return))
  previous_agent_ret <- tail(agent_return, 1)
  all_agent_mean_ret <- c(all_agent_mean_ret, agent_return)


  pair_return <- rowMeans(as.data.frame(pair_return))
  previous_pair_ret <- tail(pair_return, 1)
  all_asset_mean_ret <- c(all_asset_mean_ret, pair_return)

  i <- i + 1
}


plot.ts(all_asset_mean_ret)
plot.ts(all_agent_mean_ret)
