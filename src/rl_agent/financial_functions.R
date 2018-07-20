# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Financial functions for getting pairs trading signals 
# and evaluating strategies



#endregion
# -----------------------------------------------------------------
#region Function Definitions


ret_asset <- function(asset, previous_ret = NULL) { # returns the return of the buy&hold strategy for a single asset
  
  n <- NROW(asset)
  ret <- numeric(NROW(asset)) + 1
  
  # initiliase with previous returns if supplied
  if(!is.null(previous_ret)) ret[1] <- previous_ret
  
  for(i in 2:n) {
    ret[i] <- (( (asset[i] - asset[i - 1]) / asset[i - 1]) + 1) * ret[i - 1]
  }
  
  return(ret)
  
}

Ret <- function(pair, coef, position, cost, previous_ret = NULL, ...) {
  
  #get the number of observations
  n <- nrow(pair)
  
  #for applying transaction costs
  diff_pos <- c(0, diff(position)) # returns 1 or -1 when a position is entered
  
  #initialise vector of returns 
  additive <- numeric(NROW(pair[,1])) + 1 
  
  # initiliase with previous returns if supplied
  if(!is.null(previous_ret)) additive[1] <- previous_ret
   
  # It is tempting to think taht the return is spread/ spread(t-1) but that is not true, as (a2+b2)/(a1+ b1) /= a2/a1 + b2/b1
  # ret[i] <- (( -(spread[i] - (spread[i - 1] * (cost + 1))) / abs(spread[i - 1] * (cost + 1))) + 1) * ret[i - 1]
  # ret[i] <- (( -(spread[i] - spread[i - 1]) / abs(spread[i - 1])) + 1) * ret[i - 1] 
  
  # w poprzedniej obs wystapila zmiana (rozwazyc czy moze byc z -1 do 1) : czyli zakladamy, ze w momencie wystapienia sygnalu 
  #od razu zajmujemy pozycje. Generelnie powinnismy zajmowac tez o jeden pozniej czyli zwrot liczyc od+2 od momentu wystapienia sygnalu
  #position taken (short) #natychmiastowe zajecie pozycji? Czy nie powinno si? bada? tylko position[i-1] bo position w momencie i nie 
  #interesuje nas dla liczenia zwrotu w momencie i bo pozycja i tak zajeta
    
  
  for(i in 2:n) {
    # calculate return while being short and account for transaction cost at moment of execution
    if (position[i] == -1 & position[i - 1] != 0 & diff_pos[i - 1] != 0) {      
      additive[i] <- ((-(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]
   
    # calculate return while being short
    } else if (position[i] == -1 & position[i - 1] != 0) {      
      additive[i] <- ((-(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2] * (pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]      
   
    # calculate return while being long and account for transaction cost at moment of execution  
    } else if (position[i] == 1 & position[i - 1] != 0 & diff_pos[i -1] != 0) { #position taken (long)      
      additive[i] <- ((+(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]      
    
    # calculate return while being long
    } else if (position[i] == 1 & position[i - 1] != 0) { 
      additive[i] <- ((+(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]      
    
    # calculate return at the end of a signal  
    } else if (position[i] == 0 & position[i - 1] == 1) { #stop moment for long signal
      additive[i] <- ((+(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]      
    
    # calculate return while being short  
    } else if  (position[i] == 0 & position[i - 1] == -1) { #stop moment for short signal      
      additive[i] <- ((-(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) / (pair[i-1,1] + abs(coef[2])*pair[i-1,2])) + 1) * additive[i-1]      
    } else { #no position taken 
      additive[i] <- additive[i-1]
    }
  }
  
  #coef[1] has to be scaled to 1
 # -(coef[1]*(pair[i+1,1]-pair[i,1]) + coef[2]*(pair[i+1,2] - pair[i,2])) #effectively we are shorting when there is positive coef and long when there is neg coef
 # dodac co by wyszlo z ciaglego trzymania spreada. (od poczatku do konca long short)   
       
  return(additive)
}


# Ret () powinien miec opcje 'delay', (CO BY TEZ ULATWILO KALKULACJE?) ktora oznacz o ile miejsc przesunac positions
# w stosunku do assets. Ret traktuje positions, ze w danym momemcnie pozycja byla zajeta.
# ale w liczeniu rewardu musimy opoznic. Bo -1 ooznacza w momemcnie t zajmij pozycje i
# dopiero w t+1 mozemy liczyc zwrot.


# Plotting return
PlotAgentResult <- function(pair, spread, additive, strategy_name){

  asset1 <- ret_asset(pair[,1])
  asset2 <- ret_asset(pair[,2])  
  
  df_all <- data.frame(additive = additive, asset1 = asset1, asset2 = asset2, spread = spread) #center because cant plot geom_bar() not from the center line..
  df_all <- (df_all - 1) * 100 #percentage return with reinvested capital
  

  return.plot <- ggplot(df_all, aes(x = c(1:length(spread)))) + 
    geom_line(aes(y = additive, colour = "y1"), size = 1) + 
    geom_line(aes(y = asset1,   colour = "y2"), size = 0.8, alpha = 0.5) +
    geom_line(aes(y = asset2,   colour = "y3"), size = 0.8, alpha = 0.5) +
    scale_colour_manual("Returns",
                        values = c("y1" = "royalblue4", "y2" = "grey40", "y3" = "grey50"),
                        labels = c(strategy_name, 
                                   as.character(coint_pair[1]),
                                   as.character(coint_pair[2]))) +
    labs(y = "%RoR", xlab = 'Time Index')
  #xlab("time step") +
  #ggtitle(paste("Pair Trading of\n",coint_pair[1],"+",coint_pair[2]))
  return(return.plot)
} 

#R implementation of Differential Sharpe Ratio

diff_sharpe_ratio <- function(returns, time = NULL, eta) {
  
  
  assert_that(between(eta, 0, 1),  msg = "eta parameter must be between 0 and 1")
  assert_that(length(returns) > 0, msg = "returns array can\'t be shorter than 1")
  
  At  <- c(0) # Initalizing several vectors
  Bt  <- c(0)
  DSR <- c(0)
  #At<-At[0]
  end_idx <- length(returns) # Setting returns array size 
   # Rearranging returns array so that 0, which is at the end of raw returns array, is at the beginning and is omitted during the calculations
  start_idx <- 2
  
    
    if(end_idx==1){ return(0)}
  
    for (i in start_idx:end_idx){
      
      if(i==2) {
        if(!exists("At_temp") & !exists("Bt_temp")){
          At[i - 1] = 0
          Bt[i - 1] = 0
          DSR[i - 1] = 0
        }else{
          At[i - 1] <- At_temp
          Bt[i - 1] <- Bt_temp
        }
      }
      
      At[i] <- At[i - 1] + eta * (returns[i] - At[i - 1])
      Bt[i] <- Bt[i - 1] + eta * (returns[i] ^ 2 - Bt[i - 1])
      # Formula 28 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type=pdf
      DSR[i] <-
        (Bt[i - 1] * (At[i] - At[i - 1]) - 1 / 2 * At[i - 1] * (Bt[i] - Bt[i - 1])) /
        (Bt[i - 1] - At[i - 1] ^ 2) ^ (3 / 2) 
      
      if(is.nan(DSR[i])) DSR[i] <- 0
    }
    At_temp <<- At[end_idx]
    Bt_temp <<- Bt[end_idx]
    return(DSR)
    #[end_idx]
}

#### and C++ implementation with use of Rcpp package
library(Rcpp)
### system.time comparison

#apply(as.data.frame(x=2^seq(1,15,by=1)),1,function(x) system.time(DifferentialSharpeRatio(runif(x),eta=0.5)))
#apply(as.data.frame(x=2^seq(1,15,by=1)),1,function(x) system.time(DifferentialSharpeRatio2(runif(x),eta=0.5)))
cppFunction('

  NumericVector DifferentialSharpeRatio(NumericVector returns_, int time_=1, float eta_=0.5) {
      	NumericVector returns = returns_;
      
      	NumericVector At(returns.size());
      	NumericVector Bt(returns.size());
      	NumericVector DSR(returns.size());
      	float eta = eta_;
      	
      
      	int end_array = returns.size();
      
      	if (eta >= 0 && eta <= 1 && end_array != 0) {
      		if (end_array == 1) {
      			DSR[0] = 0;
      			return DSR;
      		}
      		else {
      			for (int i = 1; i < end_array;i++) {
      				if (i == 1) { 
      					At[i - 1] = 0;
      					Bt[i - 1] = 0;
      					DSR[i - 1] = 0;
      				}
      
      				At[i] = At[i - 1] + eta*(returns[i] - At[i - 1]);
      				Bt[i] = Bt[i - 1] + eta*(pow(returns[i], 2) - Bt[i - 1]);
      				DSR[i] = (
                        Bt[i - 1] * (At[i] - At[i - 1]) - 1.0 / 2.0 * At[i - 1] * (Bt[i] - Bt[i - 1])
                        )/ pow(Bt[i - 1] - pow(At[i - 1], 2), 3.0/2.0);
          

      				/*if( is_na(DSR[i])) {DSR[i] = 0;}*/
      			}
      
      		}
      	}
      	return DSR;
      }

')

