# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Qlearning

#moze agent moze uzywac t-1 i t, zamiast t i t+1



# -----------------------------------------------------------------
#region Dependencies



#endregion 1
# -----------------------------------------------------------------
#region Function Definition

Qcontrol <- function(pair, coef, features, Nepisodes, cost, pretrained_agent = NULL, ...){
  # Initialize all elements of the RL framework
  list[cut_points, actions, alpha , epsilon, N_0, episode, Ns, Nsa, Qsa,
       Action_track] <- InitializeRLframework(features, 'qlearning')
  # Initialize additional parameters

  params <- list(...)
  env    <- environment()
  check_for_opt_args(params, env)
  if(!exists("verbose", envir = env))  verbose <- FALSE
  if(!exists("rwrd_fun", envir = env)) rwrd_fun <- 'dsr'
  if(verbose) ns_vars(env, values = T)
    # ENDED HERE



  # If available, continue with the policy from the training period
  if(!is.null(pretrained_agent)) list[Qsa, Action_track, Nsa, Ns, epsilon, alpha, cut_points] <- pretrained_agent

  while(episode <= Nepisodes){
    i <- 1
    while (i < nrow(pair)){

      state <- StateIndexes(features[i,], cut_points) # get the current state
      action <- ChooseAction(Qsa, state, epsilon, actions) # choose action from the current state
      action_dim <- match(action, actions)
      state_action <- as.numeric(c(state, action_dim))

      Action_track <- c(Action_track, action)
      counts <- UpdateN(state, Ns, Nsa, actions, action, cut_points)
      Ns <- counts[[1]]
      Nsa <- counts[[2]]

      # Update constants
      # EPsilon should also depend on the value of the present state, could grow bigger if negative state values
      # But maybe they have been evaluated to such a value? But maybe it should grow with growing volatility or if
      # a regime switch was detected. Or otherwise: sudden changes in values indicate changing conditions and should
      # create a bigger epsilon again.
      # epsilon <- N_0 / (N_0 + Ns[matrix(state, 1)])
      epsilon <- 1/(log(Nsa[matrix(state_action, 1)])+1)
      #alpha <- 1/Nsa[matrix(state_action, 1)]
      alpha <- 1/(log(Nsa[matrix(state_action, 1)])+1)

      state_next          <- StateIndexes(features[i+1,], cut_points) # get the following state
      action_target       <- ChooseAction(Qsa, state_next, 0, actions) # deterministicaly choose the best action
      action_target_dim   <- match(action_target, actions)
      state_action_target <- as.numeric(c(state_next, action_target_dim))

      #To choose raw returns instead of Diff Sharpe Ratio it's needed to remove explicit reward_fun argument
      # For DSR reward function eta_dsr argument has to be set, otherwise the default value, 0.5, will be imposed


      rwrd <- Qreward(pair[c(i:(i + 1)), ], coef, action, cost, rwrd_fun = rwrd_fun, eta_dsr = 0.2)

      # Qlearning backup
      Qsa[matrix(state_action, 1)] <- Qsa[matrix(state_action, 1)] + alpha * (rwrd + Qsa[matrix(state_action_target, 1)] - Qsa[matrix(state_action, 1)])

      i <- i + 1
    }
   episode <- episode + 1
  }

  return(list(Qsa = Qsa, Action_track = Action_track, Nsa = Nsa, Ns = Ns,
              epsilon = epsilon, alpha = alpha, cut_points = cut_points))
}
