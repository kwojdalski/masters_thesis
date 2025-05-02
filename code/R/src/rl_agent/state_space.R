# better apply http://stackoverflow.com/questions/6827299/r-apply-function-with-multiple-parameters


UpdateStateCount <- function(df_attributes, state_count, cut_points){

  for( i in (1 : nrow(df_attributes)))
  {
    state <- StateIndexes(df_attributes[i,], cut_points)
    #State_count[matrix(rep(state, 1), nrow=1)] # dziala prawdopodobnie dlatego ze obrocil macierz zeby byla w rzedach
    state_count[t(state)] <- state_count[t(state)] + 1
  }

  return(list(state, state_count))

}

UpdateStateActionCount <- function(df_attributes, state_action_count ,actions, action, cut_points)
{
  #action - the vector of actions correspondng to the attributes
  #actions - vector of all possible actions

  for(i in (1:nrow(df_attributes))){
    action_dim <- match(action[i],actions)

    state <- StateIndexes(df_attributes[i,], cut_points)
    state <- c(state, action_dim)
    #State_count[matrix(rep(state, 1), nrow=1)] # dziala prawdopodobnie dlatego ze obrocil macierz zeby byla w rzedach
    state_action_count[t(state)] <- state_action_count[t(state)] + 1
  }

  return(list(state, state_action_count))

}

UpdateN <- function(state, Ns, Nsa ,actions, action, cut_points)
{
  #action - the vector of actions correspondng to the attributes
  #actions - vector of all possible actions
    action_dim <- match(action,actions)

    Ns[matrix(state, 1)] <- Ns[matrix(state, 1)] + 1
    Nsa[matrix(c(state,action_dim), 1)] <- Nsa[matrix(c(state,action_dim), 1)] + 1

    #State_count[matrix(rep(state, 1), nrow=1)] # dziala prawdopodobnie dlatego ze obrocil macierz zeby byla w rzedach

  return(list(Ns, Nsa))

}


StateIndexes <- function(attribute_row, cut_points) # discretizes a new row of features #unlist cutpoints before using this
{

  state_indexes <- c()

  for( i in (1:ncol(attribute_row)))
  {
    variable_order <- order(c(unlist(attribute_row[i]), unlist(cut_points[i])))[1] # returns the order of the attribute value in the set of the cut_points
    state_indexes <- c(state_indexes, variable_order)
  }

  return(state_indexes)
}

# cut - table approach

#cut(as.numeric(features[500,1]), breaks = cut_points[,1], labels = FALSE)
#cut(0.9, breaks = c(- Inf,cut_points[,1], Inf), labels = FALSE) # trzeba dodac skrajne przedzialy

#cut(attribute_row, breaks = cut_points)
#table(kk1[,1],kk1[,2], kk[,3]) # to by nam zalatwilo wsyzstko?
