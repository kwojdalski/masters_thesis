#Raw Pair Trading

Signal <- function(spread, coint_pair, m, s, TH_enter, TH_exit) { 
  #get the number of observations
  n <- NROW(spread)
  
  #initialise variables
  signal <- 0
  position <- numeric(n)
  
  
  for (i in 1:n) {  
    
    #calculate signals
    if ((spread[i] - m) > (TH_enter * s) & signal == 0) { #exceed threshhold up
      #take short position
      signal <- -1
    } else if (((m - spread[i]) > (TH_enter * s)) & signal == 0) { #exceed threshhold down
      #take long position
      signal <- 1
    } else if (((spread[i] - m) < (TH_exit * s)) & signal == -1) { #revert to mean if exit threshhold is crossed
      #exit position
      signal <- 0
    } else if (((m - spread[i]) < (TH_exit * s)) & signal == 1) { #revert to mean if exit threshhold is crossed
      #exit position
      signal <- 0
    }
    #create vector with signals
    position[i] <- signal
  }
  
  
  
  #plot spread and positions
  #barplot(position,col="blue",space = 0, border = "blue",xaxt="n",yaxt="n",xlab="",ylab="")
  #par(new=TRUE)
  #plot.ts(spread, col = "red")
  #abline(h=m, col = "grey20")
  #abline(h=m+(s*TH_enter), col = "grey60")
  #abline(h=m+(s*TH_exit), col = "grey60")
  #abline(h=m-(s*TH_exit), col = "grey60")
  #abline(h=m-(s*TH_enter), col = "grey60")
  
  
  df_all <- data.frame(position = position *max(abs(spread - m)), spread = (spread - m)) #center because cant plot geom_bar() not from the center line..
  
  spread.plot <- ggplot(df_all, aes(x = c(1:length(spread)), y = position, width = 1)) + 
    geom_bar(aes(fill = factor(position)), stat = "identity", alpha = 1) +
    scale_fill_manual(values=c("grey40", "white", "grey65"), 
                      name="Position", breaks = NULL,
                      labels=c("short", "neutral", "long")) +
    #scale_x_continuous(trans="translate3") + 
    geom_line(aes(x = c(1:length(spread)), y = spread, colour="spread"), size = 0.5) +
    scale_colour_manual(values=c("black"), name="asset") +
    ylab("spread value and position") + 
    xlab(NULL) +
    ggtitle(paste("Pair Trading of\n",coint_pair[1],"+",coint_pair[2]))
  #http://stackoverflow.com/questions/10349206/add-legend-to-ggplot2-line-plot
  
  
  
  #http://stackoverflow.com/questions/13811765/start-geom-bar-at-y0-deactivate-clip-on-only-one-side
  # zle dizala...
  
  #return vector with signals
  return (list(spread.plot,position))
}

Signal2 <- function(spread, coint_pair, m, s, TH_enter, TH_exit) { 
  #get the number of observations
  n <- NROW(spread)
  
  #initialise variables
  signal <- 0
  position <- numeric(n)
  
  
  for (i in 1:n) {  
    
    #calculate signals
    if ((spread[i] - m) > (TH_enter * s) & signal == 0) { #exceed threshhold up
      #take short position
      signal <- -1
    } else if (((m - spread[i]) > (TH_enter * s)) & signal == 0) { #exceed threshhold down
      #take long position
      signal <- 1
    } else if (((spread[i] - m) < (TH_exit * s)) & signal == -1) { #revert to mean if exit threshhold is crossed
      #exit position
      signal <- 0
    } else if (((m - spread[i]) < (TH_exit * s)) & signal == 1) { #revert to mean if exit threshhold is crossed
      #exit position
      signal <- 0
    }
    #create vector with signals
    position[i] <- signal
  }
  
  
  
  #plot spread and positions
  #barplot(position,col="blue",space = 0, border = "blue",xaxt="n",yaxt="n",xlab="",ylab="")
  #par(new=TRUE)
  #plot.ts(spread, col = "red")
  #abline(h=m, col = "grey20")
  #abline(h=m+(s*TH_enter), col = "grey60")
  #abline(h=m+(s*TH_exit), col = "grey60")
  #abline(h=m-(s*TH_exit), col = "grey60")
  #abline(h=m-(s*TH_enter), col = "grey60")
  
  
  df_all <- data.frame(position = position *max(abs(spread - m)), spread = (spread - m)) #center because cant plot geom_bar() not from the center line..
  
  spread.plot <- ggplot(df_all, aes(x = c(1:length(spread)), y = position, width = 1)) + 
    geom_bar(aes(fill = factor(position)), stat = "identity", alpha = 1) +
    scale_fill_manual(values=c("grey40", "white", "grey65"), 
                      name="Position", breaks = NULL,
                      labels=c("short", "neutral", "long")) +
    #scale_x_continuous(trans="translate3") + 
    geom_line(aes(x = c(1:length(spread)), y = spread, colour="spread"), size = 0.5) +
    scale_colour_manual(values=c("black"), name="asset") +
    ylab("spread value and position") + 
    xlab(NULL) +
    ggtitle(paste("Pair Trading of\n",coint_pair[1],"+",coint_pair[2]))
  #http://stackoverflow.com/questions/10349206/add-legend-to-ggplot2-line-plot
  
  
  
  #http://stackoverflow.com/questions/13811765/start-geom-bar-at-y0-deactivate-clip-on-only-one-side
  # zle dizala...
  
  #return vector with signals
  return (list(spread.plot,position))
}


Quarantine <- function(pair, coint_vec, position, jump = "", quar_period =""){ # sets position vector to neutral after big price shifts (if 1 period return > jump)
  
  n <- length(position)
  spread <- coint_vec[1] * pair[, 1] + coint_vec[2] * pair[, 2]
  ret_spread <- numeric(NROW(spread)) # memory allocation
  
  
  for(i in 2:n) {
    ret_spread[i] <- (( (spread[i] - spread[i - 1]) / abs(spread[i - 1])) + 1) # 1 period return of spread
  }
  
  
  for(i in 2:n) {  # sets position to neutral (no trades) if the 1 period price shift is bigger than "jump" parameter
    if(abs(1 - ret_spread[i]) > jump ){
      position[ (i) : (i + quar_period) ] <- 0    # position[i] or position[i+1]?
    }
    
  }
  
  return(position)
  
} 


#m <- mean(train_spreads[,1])
#s <- sd(train_spreads[,1])
#spread <- as.vector(test_spreads[,1])
coint_pair <- coint_pairs[,5]
coef <- coefs[,5]
spread <- train_spreads[,5]
pair <- df_currencies[base[1]:base[2], coint_pair]


m <- mean(spread)
s <- sd(spread)

sig <- Signal(spread, coint_pair, m, s, 1, 0.5)
position <- sig[[2]]
sig[1] #plot


rets <- Ret(pair,coef, spread, position, 0)
rets[2]