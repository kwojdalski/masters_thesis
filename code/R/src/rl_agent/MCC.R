# A Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Monte Carlo Control

# -----------------------------------------------------------------

#Get the list unwrap functionality
MCCbootstrap <- function(obj, ...){
  precheck_rl(obj)
  UseMethod('MCCbootstrap')
}

MCCbootstrap.CointPairs <- function(obj, ...){
  precheck_rl(coint)
  # Initialize all elements of the RL framework
  copyEnv(obj, environment())

  list[cut_points, actions, alpha , epsilon, N_0, episode, Ns, Nsa, Qsa,
       Position_track, diff, terminate] <- InitializeRLframework(features, 'mcc')

  # If available, continue with the policy from the training period
  if(!is.null(pretrained_agent)) list[Qsa, Position_track, Nsa, Ns, epsilon, alpha, cut_points] <- pretrained_agent

  # Go to first data point
  i <- 1



  while(episode < Nepisodes){
    State_track   <- data.frame() #track the states visited in a given episodes/ initialize first row to allow function is.memmber to work
    Action_track  <- c()

    state          <- StateIndexes(features[i,], cut_points) #get the next state
    action         <- ChooseAction(Qsa, state, epsilon, actions) #choose action from that state
    Position_track <- c(Position_track, action) #save action for future evaluation
    # if there was no direct jump from -1 to 1, choose the historically better action with 1-epsilon probability
    if(i == length(asset)) {action <- 0; episode <- episode + 1; print(episode)}

    # Continue to next state if an episode is not trigerred
    counts       <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
    Ns      <- counts[[1]]
    Nsa     <- counts[[2]]
    if(action == 0){
        epsilon <- N_0 / (N_0 + Ns[matrix(state, 1)])
        i <- if( i < length(asset)) i + 1 else 1
    } else { # Trigger episode

        terminate    <- FALSE
        i            <- i + 1
        State_track  <- rbind(State_track, state)
        Action_track <- c(Action_track, action)
        #needed for epsilon calculation
        Pre_Ns       <- Ns
    }

    # Loop until state is terminal
    while(!terminate){

        state          <- StateIndexes(features[i,], cut_points) #get the next state
        State_track    <- rbind(State_track, state)
        #Pre_Ns is zero in first run but that doesnt matter
        epsilon        <- N_0 / (N_0 + Pre_Ns[matrix(state, 1)])
        # choose the historically better action with 1-eps probability

        action         <- ChooseAction(Qsa, state, epsilon, actions)
        Action_track   <- c(Action_track, action)
        Position_track <- c(Position_track, action)

        #check if there is a change of position
        diff <- abs(diff(Action_track[c(length(Action_track)-1,
                                        length(Action_track))]))

        #close position if in last data point
        if( i == length(asset) && diff == 0){ diff <- 1 }

        #Backup the states after the episode
        if(diff != 0){

          # terminate if position is 0, stay in a new episode otherwise
          terminate <- if(diff == 1) TRUE else FALSE

          epi_pair <- pair[c((i - length(Action_track) + 1):i),]

          #To choose raw returns instead of Diff Sharpe Ratio it's needed to remove explicit reward_function argument
          # For DSR reward function eta_dsr argument has to be set, otherwise the default value, 0.5, will be imposed
          rwrd <- Reward(epi_pair, coef, Action_track, cost,
                         rwrd_fun = "dsr", eta_dsr = 0.8)

          # Backup States after end of the episode
          list[Action_track, State_track, Qsa, alpha, Ns, Nsa, Pre_Ns] <-
          episodeBackup(Action_track, State_track, rwrd, diff, coef,
                        cost, Qsa, alpha, Pre_Ns, Ns, Nsa)
         }

      if( i < length(asset)) {
        i <- i+1
      } else {
        i <- 1; terminate <- TRUE; action <- 0; episode <- episode + 1; print(episode)
      }
    }
 }
  return(list(Qsa = Qsa, Position_track = Position_track,
              Nsa = Nsa, Ns = Ns, epsilon = epsilon,
              alpha = alpha, cut_points = cut_points))
}


MCCbootstrap.CurPortfolio <- function(obj, ...){

  # Initialize all elements of the RL framework
  copyEnv(obj, environment())

  list[cut_points, actions, alpha , epsilon, N_0, episode, Ns, Nsa, Qsa,
       Position_track, diff, terminate] <- InitializeRLframework(features, 'mcc')

  # If available, continue with the policy from the training period
  if(!is.null(pretrained_agent)) list[Qsa, Position_track, Nsa, Ns, epsilon, alpha, cut_points] <- pretrained_agent

  # Go to first data point
  i <- 1



  while(episode < Nepisodes){
    State_track   <- data.frame() #track the states visited in a given episodes/ initialize first row to allow function is.memmber to work
    Action_track  <- c()

    state          <- StateIndexes(features[i,], cut_points) #get the next state
    action         <- ChooseAction(Qsa, state, epsilon, actions) #choose action from that state
    Position_track <- c(Position_track, action) #save action for future evaluation
    # if there was no direct jump from -1 to 1, choose the historically better action with 1-epsilon probability
    if(i == length(asset)) {action <- 0; episode <- episode + 1; print(episode)}

    # Continue to next state if an episode is not trigerred
    counts       <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
    Ns      <- counts[[1]]
    Nsa     <- counts[[2]]
    if(action == 0){
      epsilon <- N_0 / (N_0 + Ns[matrix(state, 1)])
      i <- if( i < length(asset)) i + 1 else 1
    } else { # Trigger episode

      terminate    <- FALSE
      i            <- i + 1
      State_track  <- rbind(State_track, state)
      Action_track <- c(Action_track, action)
      #needed for epsilon calculation
      Pre_Ns       <- Ns
    }

    # Loop until state is terminal
    while(!terminate){

      state          <- StateIndexes(features[i,], cut_points) #get the next state
      State_track    <- rbind(State_track, state)
      #Pre_Ns is zero in first run but that doesnt matter
      epsilon        <- N_0 / (N_0 + Pre_Ns[matrix(state, 1)])
      # choose the historically better action with 1-eps probability

      action         <- ChooseAction(Qsa, state, epsilon, actions)
      Action_track   <- c(Action_track, action)
      Position_track <- c(Position_track, action)

      #check if there is a change of position
      diff <- abs(diff(Action_track[c(length(Action_track)-1,
                                      length(Action_track))]))

      #close position if in last data point
      if( i == length(asset) && diff == 0){ diff <- 1 }

      #Backup the states after the episode
      if(diff != 0){

        # terminate if position is 0, stay in a new episode otherwise
        terminate <- if(diff == 1) TRUE else FALSE

        epi_pair <- pair[c((i - length(Action_track) + 1):i),]

        #To choose raw returns instead of Diff Sharpe Ratio it's needed to remove explicit reward_function argument
        # For DSR reward function eta_dsr argument has to be set, otherwise the default value, 0.5, will be imposed
        rwrd <- Reward(epi_pair, coef, Action_track, cost,
                       rwrd_fun = "dsr", eta_dsr = 0.8)

        # Backup States after end of the episode
        list[Action_track, State_track, Qsa, alpha, Ns, Nsa, Pre_Ns] <-
          episodeBackup(Action_track, State_track, rwrd, diff, coef,
                        cost, Qsa, alpha, Pre_Ns, Ns, Nsa)
      }

      if( i < length(asset)) {
        i <- i+1
      } else {
        i <- 1; terminate <- TRUE; action <- 0; episode <- episode + 1; print(episode)
      }
    }
  }
  return(list(Qsa = Qsa, Position_track = Position_track,
              Nsa = Nsa, Ns = Ns, epsilon = epsilon,
              alpha = alpha, cut_points = cut_points))
}



















QsaEvaluate <- function(Qsa, State_track, Action_track, actions, Ns, Nsa, rwrd){
  # Monte Carlo Qsa update of the states visited in an Episode
  # (every visit evaluation)

  for (i in 1:nrow(State_track)){
    action <- Action_track[i]
    state <- as.numeric(State_track[i,])

    action_dim <- match(action, actions)
    state_action <- as.numeric(c(state, action_dim))

    counts <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
    Ns  <- counts[[1]]
    Nsa <- counts[[2]]

    alpha <- 1 / Nsa[matrix(state_action, 1)]
    Qsa[matrix(state_action, 1)] <- Qsa[matrix(state_action, 1)] + alpha*(rwrd - Qsa[matrix(state_action, 1)])
  }

  return(list(Ns, Nsa, Qsa))
}





#endregion
# -----------------------------------------------------------------
#Test run and Result analysis



#parametry przeniesc do configu
#te wywolania ponizej do foldera run
#Poczatek MMC do osobnej funkcji
#Take plotting out of the financial functions
#dodac sharpe ratio do wykresow
#correct 0s may be mixed with negative subscripts error

# duzy skok ma wplyw na cala policy.
