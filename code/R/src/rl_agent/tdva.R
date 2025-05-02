# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Qlearning

#moze agent moze uzywac t-1 i t, zamiast t i t+1



# -----------------------------------------------------------------
#region Dependencies

#endregion
# -----------------------------------------------------------------
#region Function Definition

tdva_control <- function(pair, coef, features, Nepisodes, cost, pretrained_agent=NULL, ...){
  # Initialize all elements of the RL framework
  params <- list(...)

  list[actions, f_params, alpha, epsilon, N_0, episode, Action_track] <- InitializeRLframework(features, 'tdva')

  # If available, continue with the policy from the training period
  if(!is.null(pretrained_agent)) list[actions, f_params, alpha, epsilon, N_0, episode, Action_track] <- pretrained_agent

  while(episode <= Nepisodes)
  {
   i <- 1
    while (i < nrow(pair))
    {

    state <- features[i,]
    action <- ChooseActionTD(f_params, state, epsilon, actions) # choose action accordin to the present state of FA (separate model for each action)
    action_dim <- match(action, actions)
    Qsa <- c(f_params %*% t(as.matrix(state)))[action_dim]
    state_action <- as.numeric(c(state, action_dim))

    Action_track <- c(Action_track, action)
    Action_count <- table(Action_track)
    a_count = Action_count[names(Action_count) == action]

    # Update constants
    # EPsilon should also depend on the value of the present state, could grow bigger if negative state values
    # But maybe they have been evaluated to such a value? But maybe it should grow with growing volatility or if
    # a regime switch was detected. Or otherwise: sudden changes in values indicate changing conditions and should
    # create a bigger epsilon again.
    epsilon <- N_0 / (N_0 + i)
    #alpha <- 1/Nsa[matrix(state_action, 1)]
    alpha <- N_0 / (N_0 + a_count)

    state_next <- features[i+1,] # get the following state
    action_target <- ChooseActionTD(f_params, state_next, epsilon, actions) # choose the next action with the same policy (actualy should also use old epsilon)
    action_target_dim <- match(action_target, actions)
    Qsa_target <- c(f_params %*% t(as.matrix(state)))[action_target_dim]
    state_action_target <- as.numeric(c(state_next, action_target_dim))

    # 1-step return
    ret <- Qreward(pair[c(i:(i+1)),], coef, action, cost)

    # Update the model
    delta = unlist(alpha*(ret + Qsa_target - Qsa)*state)
    f_params[action_dim,] <- f_params[action_dim,] + delta


    i <- i + 1
    }
   episode <- episode + 1
  }

  return(list(Qsa, Action_track, Nsa, Ns, epsilon, alpha, cut_points))
}
