# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Cointegrations


# -----------------------------------------------------------------
#region Dependencies

# Cointegration
p_load(urca, tseries)

#endregion
# -----------------------------------------------------------------
#region Function Definitions


# GetCOintPairs() inputs a data frame with timeseries data in columns and
# returns a data frame where each column includes the names of cointegrated pairs

# uwaga: w notatkach z StackExchange widac, ze moze nam to rownoczesnie wykryc stacjonarne szeregi. Moze uzyc tego?

GetCointPairs <- function(df_currencies, c_currencies, learn_period, cval) { 
  
  # returns a df where each column includes names of the cointegrated series based on data df_currencies
  # cval (1,2,3) is the critical value - (10 percentile,5pct,1pct)
  
  # Cut the data to the specified learn period
  df_currencies <- df_currencies[c(learn_period[1]:learn_period[2]),]
  # all 2-element combinations of the given currency pair names
  df_combn <- combn(c_currencies, 2) 
  rejected_combn <- numeric()
  assert_that(NCOL(df_combn) > 0, msg = 'df_combn can\'t be empty')
  # find the ones that are not cointegrated with the specified p-value  
  for( i in 1 : NCOL(df_combn)) {
    
    if( ca.jo(df_currencies[df_combn[,i]])@teststat[2] < ca.jo(df_currencies[df_combn[,i]])@cval[2,cval]) { 
      # jezeli odrzucamy na poziomi 0.95 to skreslamy kombinacj? par z dalszych rozwazan//wypisac df_curr jako argument? najpierw trzeba wywolac df currencies. head(df_currencies[combn[,1]]) dzia?a.
      rejected_combn <- c(rejected_combn, i) #colnames(df_combn) # saves columns of uncointegrated pairs
    } 
  }
  df_combn %<>% as.data.frame() %>% select(-rejected_combn)
    #subset w ponizszej funkcji zle dziala przy pustym wektorze w subset dlatego takie IF...
  return(df_combn)
}

CointSpreads <- function(coint_pairs, df_currencies, learn_period, test_period){
  
  # returns the cointegration spread for pairs of known cointegrated series
  # devides this spread into train/test, where test 
  # has the cointegration coefficients from the train period
   
  l_sets <- list()
  
  df_train <- df_currencies[c(learn_period[1]: learn_period[2]),]
  df_test <- df_currencies[c(test_period[1]: test_period[2]),]
  df_all <- df_currencies
  
  for( i in (1 : (ncol(coint_pairs)))){
    pair <- as.character(coint_pairs[, i])
    
    train_sample <- df_train[,pair]
    test_sample <- df_test[, pair]
    all <- df_all[, pair]
    
    coint_vec <- ca.jo(train_sample)@V[,1]    # coint coefs from the learning period / cointegration coefficients for the given pair in the first period / wouldn' it be better to return this already in GetCointPairs() ?
    train_spread <- coint_vec[1] * train_sample[, 1] + coint_vec[2] * train_sample[, 2]
    test_spread <-  coint_vec[1] * test_sample[, 1] + coint_vec[2] * test_sample[, 2]
    all_spread <-   coint_vec[1] * df_all[, 1] + coint_vec[2] * df_all[, 2]
    
    l_sets[["train"]][[paste0(coint_pairs[1,i],"+", coint_pairs[2, i])]] <- train_spread
    l_sets[["test"]][[paste0(coint_pairs[1,i], "+", coint_pairs[2, i])]] <- test_spread
    l_sets[["all"]][[paste0(coint_pairs[1,i],  "+", coint_pairs[2, i])]] <- all_spread
    l_sets[["coef"]][[paste0(coint_pairs[1,i], "+", coint_pairs[2, i])]] <- coint_vec
  }    
  
  return(l_sets)
  
}

GetPairs <- function(df_currencies, c_currencies, learn_period){
  
  df_train <- df_currencies[c(learn_period[1]: learn_period[2]),]
  df_test  <- df_currencies[c(test_period[1]: test_period[2]),]
  df_all   <- df_currencies
  l_sets   <- alply(c_currencies, 1, function(cur){
    return(list(train = df_train[, cur],
                test  = df_test[,  cur],
                all   = df_all[,   cur]))
  })
  names(l_sets) <- c_currencies
  return(l_sets)
}





