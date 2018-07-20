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
test_period  <- c(1001, 3000)

quotes <- df_currencies[, 1:3]


train_quotes  <- as.data.frame(quotes[train_period[1]:train_period[2], ])
test_quotes   <- as.data.frame(quotes[test_period[1]:test_period[2], ])
train_test_quotes   <- llply(list(train_period, test_period), 
                             .fun = function(x) seq.int(from = x[1], to = x[2])) %>% 
  flatten_int() %>% {as.data.frame(quotes[., ])}
                             

#endregion
# -----------------------------------------------------------------
#region Attributes extraction

#Create the data frame of attributes for all cointegrated pairs

i_delay <- 2
l_attributes <- AttributesAll(train_test_quotes, colnames(train_test_quotes), i_delay)

#endregion
# -----------------------------------------------------------------
#region RL backtest data preparation

# Pick one of the pairs and it's attributes

pair_idx   <- 1
train_test_quotes


train_pair     <- df_currencies[c(train_period[1]:train_period[2]), 'EURGBP']
train_quotes   <- train_quotes[[pair_idx]]
train_features <- as.data.frame(l_attributes[pair_idx])[c(train_period[1]:train_period[2]),]


test_pair     <- df_currencies[c(test_period[1]:test_period[2]), 'EURGBP']
test_quotes   <- test_quotes[[pair_idx]]
test_features <- as.data.frame(l_attributes[pair_idx])[c(test_period[1]:test_period[2]),]


l_attributes[pair_idx][[1]] %>% head()
train_quotes %>% {data.frame(x=.)} %>%  {ggplot(., aes(seq(1:nrow(.)), y = .))+ geom_line()}
test_quotes %>% {data.frame(x=.)} %>%  {ggplot(., aes(seq(1:nrow(.)), y = .))+ geom_line()}



#endregion
# -----------------------------------------------------------------
#region MCC test
dim(train_features)
Nepisodes     <- 10
cost          <- 0
cur_portfolio <- CurPortfolio$new(train_quotes,
                            mutate(train_features, r = seq_len(nrow(train_features))),
                            Nepisodes, cost,
                            algorithm = 'mcc')
pretrained_agent <- cur_portfolio$train()

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

#experience replay - nie przejmuje sie co sie dzieje do moment uduzego skoku, a potem juz tylko idzie w gore.
pretrained_agent <- Qcontrol(train_pair, coef, train_features, Nepisodes, cost, verbose = T, rwrd_fun = 'dsr')
pretrained_pos   <- pretrained_agent[[2]]
names(pretrained_agent$Qsa)
(v <- structure(10*(5:8), names = LETTERS[1:4]))
f2 <- function(x, y) outer(rep(x, length.out = 3), y)
(a2 <- sapply(v, f2, y = 2*(1:5), simplify = "array"))

as.list(pretrained_agent$Qsa)
pretrain_return  <- Ret(train_pair, coef, tail(pretrained_pos, nrow(train_pair)), 0)
PlotAgentResult(train_pair, train_spread, pretrain_return, 'Qlearning Control')


tested_agent <- Qcontrol(test_pair, coef, test_features, 1, cost, pretrained_agent)
test_pos <- tested_agent[[2]]
test_return <- Ret(test_pair, coef, tail(test_pos, nrow(test_pair)), 0)
PlotAgentResult(test_pair, test_spread, test_return, 'Qlearning Control')


