source('./src/rl_agent/pre_run.R')

#c_currencies <- list.files("C:/Users/user/Desktop/UCL final project/data/1min") # 1min data
c_currencies <- c("EURGBP", "EURUSD", "USDJPY", "EURJPY", "CHFJPY", "GBPUSD", "AUDJPY", "EURCAD")
#c_currencies <- randomObs(c_currencies, 12) # 12 randon currencies
#c_currencies <- c_currencies[! c_currencies %in% c(c_currencies1)]
#df_currencies <- GetQuotes(c_currencies, data_path, ".csv")

# New data 08.2015
df_currencies <- GetQuotes(c_currencies, data_path, ".csv")

#endregion
# -----------------------------------------------------------------
#region Cointegration extraction

# Get all cointegrated pairs on a specified base interval

train_period <- c(1, 1000)
test_period  <- c(1001, 300)

coint_pairs <- GetCointPairs(df_currencies, c_currencies, train_period, 3)
coint_spreads <- CointSpreads(coint_pairs, df_currencies, train_period, test_period)

train_spreads <- as.data.frame(coint_spreads[[1]])
test_spreads  <- as.data.frame(coint_spreads[[2]])
all_spreads   <- as.data.frame(coint_spreads[[3]])
coefs         <- as.data.frame(coint_spreads[[4]])


#endregion
# -----------------------------------------------------------------
#region Attributes extraction

#Create the data frame of attributes for all cointegrated pairs

i_delay <- 2
l_attributes <- AttributesAll(all_spreads, colnames(all_spreads), i_delay)


#endregion
# -----------------------------------------------------------------
#region RL backtest data preparation

# Pick one of the pairs and it's attributes

pair_ind   <- 1
coint_pair <- coint_pairs[, pair_ind]
coef       <- coefs[, pair_ind]

train_pair     <- df_currencies[c(train_period[1]:train_period[2]), coint_pair]
train_spread   <- train_spreads[[pair_ind]]
train_features <- as.data.frame(l_attributes[pair_ind])[c(train_period[1]:train_period[2]),]

test_pair     <- df_currencies[c(test_period[1]:test_period[2]), coint_pair]
test_spread   <- test_spreads[[pair_ind]]
test_features <- as.data.frame(l_attributes[pair_ind])[c(test_period[1]:test_period[2]), ]
l_attributes[pair_ind][[1]] %>% head()
train_spread %>% {data.frame(x=.)} %>%  {ggplot(., aes(seq(1:nrow(.)), y = .))+ geom_line()}
test_spread %>% {data.frame(x=.)} %>%  {ggplot(., aes(seq(1:nrow(.)), y = .))+ geom_line()}



#endregion
# -----------------------------------------------------------------
#region MCC test
dim(train_features)
Nepisodes <- 10
cost      <- 0
coint     <- CointPairs$new(train_spread, train_pair, coef, 
                            train_features,
                            # train_features,
                            Nepisodes, cost, algorithm = 'mcc')

pretrained_agent <- MCCbootstrap(coint)
pretrained_pos   <- pretrained_agent[[2]]
pretrain_return  <- Ret(train_pair, coef, tail(pretrained_pos, nrow(train_pair)), cost = cost)
PlotAgentResult(train_pair, train_spread, pretrain_return, 'Monte Carlo Control')

coint_test   <-  CointPairs$new(test_spread, test_pair, coef, 
                                mutate(test_features, r = seq_len(nrow(test_features))),
                                Nepisodes, cost, pretrained_agent = pretrained_agent)
tested_agent <-  MCCbootstrap(coint_test)
test_pos     <-  tested_agent[[2]]

test_return  <-  Ret(test_pair, coef, tail(test_pos, nrow(test_pair)), 0)
PlotAgentResult(test_pair, test_spread, test_return, 'Monte Carlo Control')

# PRZEKAZANIE VSA DO TESTOWEGO OKRESU. Jak to ma jakies parametry typu alpha to to powinniem byc hyper parametr
# jezeli mamy zamiar ogolnie optymalizowac. Czy dlugosc epizodu i mierzenia kointegracji (base i test period) mogla 
# by byc wstawiona jako parametry dla agenta?

#endregion
# -----------------------------------------------------------------
#region Qlearning test

Nepisodes <- 5
cost      <- 0

pretrained_agent <- Qcontrol(train_pair, coef, train_features, Nepisodes, cost, verbose = T, rwrd_fun = 'dsr')
pretrained_pos   <- pretrained_agent[[2]]

pretrain_return  <- Ret(train_pair, coef, tail(pretrained_pos, nrow(train_pair)), 0)
PlotAgentResult(train_pair, train_spread, pretrain_return, 'Qlearning Control')


tested_agent <- Qcontrol(test_pair, coef, test_features, 1, cost, pretrained_agent)
test_pos <- tested_agent[[2]]
test_return <- Ret(test_pair, coef, tail(test_pos, nrow(test_pair)), 0)
PlotAgentResult(test_pair, test_spread, test_return, 'Qlearning Control')


