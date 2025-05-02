ChooseAction <- function(Qsa, state_index, epsilon, actions, ...){

  # epsilon-Greedy choice of action given a state, present Q(s,a) values and
  # epsilon

  # this terrible peace of code is necessary because R doesnt subset arrays by dimensions...
  # Get action-values for each of the three actions
  vals <- c()
  for (i in 1 : length(actions)){
    state_action <- c(state_index,i)
    vals[i] <- Qsa[matrix(state_action, 1)]
  }
  # The greedy action
  greedy <- actions[which.max(vals)]
  probs = rep(epsilon/length(actions), length(actions))
  probs[which(greedy == actions)] <- probs[which(greedy == actions)] + 1 - epsilon
  action <- sample(actions, 1, prob = probs)
  # Epsilon-greedy choice of the actual action

  return(action)
}

ChooseActionTD <- function(f_params, state, epsilon, action)
{
  action_values <- f_params %*% t(as.matrix(state)) # values of each action according to the present state of the FA

  # The greedy action
  greedy <- actions[which.max(action_values)]

  # Epsilon-greedy choice of the actual action
  probs = rep(epsilon/length(actions), length(actions))
  probs[which(greedy == actions)] <- probs[which(greedy == actions)] + 1-epsilon
  action <- sample(actions, 1, prob = probs)


  return(action)

}

InitializeRLframework <- function(features, rlalgo)
{
  if (rlalgo == 'mcc'){
    cut_points <- apply(features, 2, CreatePeriods, i_periodCount = 1,
                        ch_method = "freq", i_multiplier = 1,
                        include_extreme = FALSE) # Create cut-points for a whole df

    cut_points <- as.data.frame(cut_points)
    #cut_points is from the same period as the train set (not possible in trading practice)

    N_0        <- 10 #parameter for epsilon_t computation
    actions    <- c(-1,0,1) #short, neutral, long
    dimensions <- c(1 + apply(cut_points, 2, length), length(actions)) # count how many cut point there are (dimensions = cutpoints + 1)
    Ns         <- array(0, dim = dimensions[c(-length(dimensions))])
    Nsa        <- array(0, dim = dimensions)
    Qsa        <- Nsa #state-action function

    Position_track <- c() #save all actions for future evaluation
    episode <- 1   #initialize episode counter
    epsilon <- 1   #initialize epsilon to 1
    alpha   <- 1
    i       <- 1   #data point counter
    diff    <- 0   #initialize difference between last two actions
    terminate <- TRUE

    return(list(cut_points = cut_points, actions = actions,
                alpha = alpha , epsilon = epsilon,
                N_0 = N_0, episode = episode,
                Ns = Ns, Nsa = Nsa, Qsa = Qsa,
                Position_track = Position_track,
                diff = diff, terminate = terminate))
  }

  if (rlalgo == 'qlearning'){
    cut_points <- apply(features, 2, CreatePeriods, i_periodCount = 1, ch_method = "freq", i_multiplier = 1, include_extreme = FALSE) # Create cut-points for a whole df
    cut_points <- as.data.frame(cut_points)

    N_0 <- 10 #parameter for epsilon_t computation
    actions <- c(-1,0,1) #short, neutral, long
    dimensions <- c(1 + apply(cut_points,2, length), length(actions)) # count how many cut point there are (dimensions = cutpoints + 1)
    Ns <- array(0, dim = dimensions[c(-length(dimensions))])
    Nsa <- array(0, dim = dimensions)
    Qsa <- Nsa #state-action function

    State_track <- data.frame() #track the states visited in a given episodes/ initialize first row to allow function is.memmber to work
    Action_track <- c()

    episode <- 1 #initialize episode counter
    epsilon <- 1 #initialize epsilon to 1
    alpha <- 1
    i <- 1 #data point counter

    return(list(cut_points = cut_points, actions = actions, alpha = alpha ,
                epsilon = epsilon, N_0 = N_0, episode =episode, Ns = Ns,
                Nsa = Nsa, Qsa = Qsa, Action_track = Action_track))
  }

  if (rlalgo == 'tdva'){

    N_0 <- 10 #parameter for epsilon_t computation
    actions <- c(-1,0,1) #short, neutral, long
    f_params <- matrix(data = numeric(ncol(features)*length(actions)),ncol=ncol(features), nrow=length(actions)) #params for v_f representing each action

    Action_track <- c()

    episode <- 1 #initialize episode counter
    epsilon <- 1 #initialize epsilon to 1
    alpha <- 1
    i <- 1 #data point counter

    return(list(actions, f_params, alpha, epsilon, N_0, episode, Action_track))
  }
}


precheck_rl <- function(obj){
  assert_that('features' %in% names(obj),   msg = 'Object must have features object')
  assert_that(is.data.frame(obj$features),  msg = 'Object must have features of data.frame type')
  assert_that(length(unique(map_chr(obj$features, class))) >=1,
              msg = 'Object must have only numeric values in features data.frame')

  unique_val_features <- map_int(obj$features, .f = function(x) length(unique(x)))
  assert_that(!1 %in% unique_val_features,
              msg = glue("The following columns in features data.frame doesn\'t have more than one unique values:
                         {coint$features %>% select(which(1==unique_val_features)) %>%
                           colnames() %>% paste0(collapse =', ')}"))



}
