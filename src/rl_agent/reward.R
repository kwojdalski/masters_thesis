Reward <- function(pair, coef, Action_track, cost,rwrd_fun='raw_return', eta_dsr=0.5){

  '
  Returns the Return for each (s,a) pair in the episode.
  The Return is part of the RL framework and is defined
  as the sum of the rewards in an episode.
  
  Input:
  
    pair - data frame containing price series of both assets
    of length equal to the length of the episode
    
    coef - cointegrating coefficient for the assets in the pair
    
    Action_track - vector of actions carried out in the episode
    
    cost - cost of engaging in a trade (changing action)
  '
  
  # get the number of observations
  n <- NROW(pair)
  
  # initialise vector of returns. The reward is defined as 
  # percentage return
  additive <- numeric(NROW(pair[,1]))
  
  
    # reward in the last visited state interval is 0
    # therefor up to n-1
    for(i in 1:(n-1)) {
      
      # calculate return of the paird following each (s,a) pair in the episode.
      # An episode consists only of a=1 or a=-1 and a final different action which
      # termiantes the episode.
      
      if (Action_track[i] == -1) {      
        additive[i] <- (-(coef[1]*(pair[n,1]-pair[i,1]) + coef[2]*(pair[n,2] - pair[i,2])) / (pair[i,1] + abs(coef[2])*pair[i,2]))
        
      } else if (Action_track[i] == 1) { #Action_track taken (long)      
        additive[i] <- (coef[1]*(pair[n,1]-pair[i,1]) + coef[2]*(pair[n,2] - pair[i,2])) / (pair[i,1] + abs(coef[2])*pair[i,2])     
      }
    }
  
  if(rwrd_fun == "dsr"){
    additive <-c(additive[length(additive)], additive[1:(length(additive) - 1)])
    dsr <- diff_sharpe_ratio(additive, eta = eta_dsr)
    reward <- c(dsr[2:length(dsr)], dsr[1])
  }else{
    reward <- additive
  }
  
  return(reward)  
}

Qreward <- function(pair, coef, Action_track, cost, rwrd_fun = 'raw_return', eta_dsr=0.5){
  
  '
  Returns the Return for each (s,a) pair in the episode.
  The Return is part of the RL framework and is defined
  as the sum of the rewards in an episode.
  
  Input:
  
  pair - data frame containing price series of both assets
  of length equal to the length of the episode
  
  coef - cointegrating coefficient for the assets in the pair
  
  Action_track - vector of actions carried out in the episode
  
  cost - cost of engaging in a trade (changing action)
  '
  
  # get the number of observations
  n <- nrow(pair)
  
  # initialise vector of returns. The reward is defined as 
  # percentage return
  additive <- numeric(NROW(pair[,1]) - 1)
  
  # reward in the last visited state interval is 0
  # therefor up to n-1
  for(i in 1:(n-1)) {
    
    # calculate return of the paird following each (s,a) pair in the episode.
    # An episode consists only of a=1 or a=-1 and a final different action which
    # termiantes the episode.
    if (Action_track[i] == -1) {      
      additive[i] <- (-(coef[1]*(pair[n,1]-pair[i,1]) + coef[2]*(pair[n,2] - pair[i,2])) / (pair[i,1] + abs(coef[2])*pair[i,2]))       
    } else if (Action_track[i] == 1) { #Action_track taken (long)      
      additive[i] <- (coef[1]*(pair[n,1]-pair[i,1]) + coef[2]*(pair[n,2] - pair[i,2])) / (pair[i,1] + abs(coef[2])*pair[i,2])     
    } else if (Action_track[i] == 0) {
      additive[i] <- 0
    }
    
  }
  rwrd <- additive
  
  
  if(rwrd_fun == "dsr") {
    additive <- c(0, additive)
    dsr <- diff_sharpe_ratio(additive, eta = eta_dsr)
    rwrd <- c(dsr[2:length(dsr)])
  }
  
  return(rwrd)  
}

episodeBackup <- function(Action_track, State_track, rwrd, diff, coef, cost, Qsa, alpha, Pre_Ns, Ns, Nsa){
  
  actions <- c(-1,0,1) #short, neutral, long
  
  # Episode terminated with a 0
  if(diff == 1){
                   
    for (l in 1:nrow(State_track))
    {
      action <- Action_track[l]
      state <- as.numeric(State_track[l,])
      
      action_dim <- match(action, actions)
      state_action <- as.numeric(c(state, action_dim))
      
      counts <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
      Ns <- counts[[1]]  
      Nsa <- counts[[2]]
      
      #alpha <- 1/Nsa[matrix(state_action, 1)]
      alpha <- 1/(log(Nsa[matrix(state_action, 1)])+1)
      # MCC backup
      Qsa[matrix(state_action, 1)] <- Qsa[matrix(state_action, 1)] + alpha*(rwrd[l] - Qsa[matrix(state_action, 1)])        
    }
  }
  
  if(diff == 2)
  {   
    for (j in 1:nrow(State_track))
    {
      action <- Action_track[j]
      state <- as.numeric(State_track[j,])
      
      action_dim <- match(action, actions)
      state_action <- as.numeric(c(state, action_dim))
      
      counts <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
      Ns <- counts[[1]]  
      Nsa <- counts[[2]]
      
      #alpha <- 1/Nsa[matrix(state_action, 1)]
      alpha <- 1/(log(Nsa[matrix(state_action, 1)])+1) 
      Qsa[matrix(state_action, 1)] <- Qsa[matrix(state_action, 1)] + alpha*(rwrd[j] - Qsa[matrix(state_action, 1)])
    }
    State_track <- State_track[nrow(State_track),]
    Action_track <- Action_track[length(Action_track)]
    # Epsilon updates need to use the new
    Pre_Ns <- Ns
  }
 
  return(list(Action_track, State_track, Qsa, alpha, Ns, Nsa, Pre_Ns))
}


save <- function(pair, coef, spread, Action_track, cost) {
  
  # standard percentage return   
  rwrd <- Ret(pair, coef, spread, Action_track, cost, FALSE)
  rwrd <- rwrd[[1]]
  rwrd <- rwrd[-c(1)] - 1 # back to percentage return
  
  Ret2(pair, coef, spread, Action_track, cost)
  
  #     #get the number of observations
  #     n <- NROW(spread)
  #     
  #     #initialise vector of returns
  #     additive <- numeric(NROW(pair[,1]))
  #     multiplicative <- numeric(NROW(pair[,1])) + 1 
  #     multiplicative[1] <- 1 - cost #cost for opening Action_track
  #        
  #     for(i in 2:n) {
  #       if (Action_track == -1) { 
  #         multiplicative[i] <- ((-(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) /
  #                            (pair[i-1,1] + abs(coef[2])*pair[i-1,1])) + 1) * multiplicative[i-1]  
  #         additive[i] <- (-(coef[1]*(pair[n,1]-pair[i-1,1]) + coef[2]*(pair[n,2] - pair[i-1,2])) /
  #                                  (pair[i-1,1] + abs(coef[2])*pair[i-1,1]))  
  #       } else if (Action_track == 1) { #Action_track taken (long)
  #         multiplicative[i] <- ((+(coef[1]*(pair[i,1]-pair[i-1,1]) + coef[2]*(pair[i,2] - pair[i-1,2])) /
  #                            (pair[i-1,1] + abs(coef[2])*pair[i-1,1])) + 1) * multiplicative[i-1] 
  #         additive[i] <- (+(coef[1]*(pair[n,1]-pair[i-1,1]) + coef[2]*(pair[n,2] - pair[i-1,2])) /
  #                            (pair[i-1,1] + abs(coef[2])*pair[i-1,1]))
  #       }
  #     }
  #     
  #     #include the cost
  #     multiplicative[n] <- multiplicative[n] * (1-cost)
  #     additive <- additive - cost*2
  #     
  #     additive is the actual reward vector for an episode of MCC
  #     return(list(additive, multiplicative, multiplicative[n]))
  return(rwrd)
}  