# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Discretize the State Space


# -----------------------------------------------------------------
#region Dependencies


#install.packages("discretization")



#endregion
# -----------------------------------------------------------------
#region Function Definitions

#CreatePeriods create boundaries that differs states. Otherwise, if we haven't used them, you
# would need to perform value approximation for states.

CreatePeriods <- function(c_var, i_periodCount, ch_method, i_multiplier, include_extreme){

  c_periods <- c()
  c_tmpvar  <- na.omit(as.vector(c_var))
  median    <- median(c_tmpvar)


  if(ch_method == "SD"){
    sd <- i_multiplier * (sd(c_tmpvar))
    mean <- mean(c_tmpvar)

    c_periods <- sapply(1 : i_periodCount, function(i) mean + i * sd)
    c_periods <- c(c_periods, mean, sapply(1 : i_periodCount, function(i) mean - i * sd))
    c_periods <- sort(c_periods)

  } else if(ch_method == "freq") {
    c_below <- sort(c_var[which(c_var <= median)], decreasing=TRUE)
    c_above <- sort(c_var[which(c_var > median)])
    i_belowStep <- floor(length(c_below) / i_periodCount)
    i_aboveStep <- floor(length(c_above) / i_periodCount)
    c_periods <- numeric(i_periodCount * 2)

    for (i in 1 : i_periodCount) {
      c_periods[i]                  <- i_multiplier * c_below[i * i_belowStep]
      c_periods[i_periodCount + i]  <- i_multiplier * c_above[i * i_aboveStep]
    }
    c_periods <- c(c_periods,median)
    c_periods <- sort(c_periods)
  }
  # remove the cutpoints for above extreme values
  if(!include_extreme){ c_periods <- c_periods[-c(1,length(c_periods))]}

  c_periods <- as.data.frame(c_periods)
  return(c_periods)
}

Discretize <- function(c_data, c_periods)
{
  i_countPeriods <- length(c_periods)
  c_periods <- sort(c_periods)
  c_result <- numeric(length(c_data))
  half <- (i_countPeriods - 1) / 2

  if(i_countPeriods == 1){
    c_result[c_data <= c_periods[1]] <- -1
    c_result[c_data > c_periods[1]] <- 1
    c_result[is.na(c_data)] <- NA
  } else {
    i <- 1
    while( i <= length(c_data)){
      while(is.na(c_data[i]))
      {
        c_result <- c(c_result, NA)
        i <- i + 1
      }
      if(c_data[i] <= c_periods[1]) c_result <- c(c_result , -(i_countPeriods - 1) /2 - 1 )
      if(c_data[i] > c_periods[ i_countPeriods ] ) c_result <- c(c_result, (i_countPeriods - 1) /2 + 1)

      for(j in (1 : half)){
        if((c_periods[j] < c_data[i])  && (c_data[i] <= c_periods[j + 1])){
          c_result <- c(c_result , -half + j - 1 )
          break
        }
      }

      for(j in ((half + 2) : i_countPeriods - 1 )){
        if((c_periods[j] < c_data[i])  && (c_data[i] <= c_periods[j + 1])){
          c_result <- c(c_result , -half + j)
          break
        }
      }
      #c_data[i]
      i <- i + 1

      #c_result
    }
  }
  return(c_result)
}

###

#endregion
# -----------------------------------------------------------------
#region Other Discretization Methods

#using methods from library(Discretization)
#discrete <- chiM(features[101:200,], alpha = 0.05) # don't know how it reacts to many NAs in the first rows
#discretized_features <- discrete$Disc.data
#cut_points <- as.data.frame(discrete$cutp)

#using own functions
#cut_points <- apply(features, 2, CreatePeriods, i_periodCount = 1, ch_method = "freq", i_multiplier = 1, include_extreme = FALSE) # Create cut-points for a whole df
#cut_points <- as.data.frame(cut_points)
#discretized_features <- apply(features, 2, Discretize, c_periods = cut_points) # doesnt work properly?

f <- function(i) {
  Discretize(features[,i], cut_points[,i])
}

#discretized_features <- sapply(c(1:ncol(features)), function(i) f(i), USE.NAMES = TRUE)
#colnames(discretized_features) <- colnames(features)

#table(as.data.frame(Discretize(features[,3], cut_points[,3]))) # compare to check if it works
#table(discretized_features[,3]) # it works
