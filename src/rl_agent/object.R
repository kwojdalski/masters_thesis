CointPairs <- R6Class("CointPairs",
                      inherit = CurPortfolio,
                      public = list(
                        asset     = NULL,
                        pair      = NULL,
                        coef      = NULL,
                        features  = NULL,
                        Nepisodes = NULL,
                        cost      = NULL,
                        pretrained_agent = NULL,
                        algorithm        = NULL,
                        
                        initialize = function(asset, pair, coef, features, 
                                              Nepisodes, cost, pretrained_agent = NULL,
                                              algorithm = NULL) {
                          
                          self$asset <- asset
                          self$pair <- pair
                          self$coef <- coef
                          self$features <- features
                          self$Nepisodes <- Nepisodes
                          self$cost <- cost
                          assert_that(is.null(pretrained_agent) || dim(self$features)[2] == dim(pretrained_agent$cut_points)[2],
                                      msg = glue('pretrained agent has a different number of features {dim(self$features)[2]} than the number of features for state space {dim(pretrained_agent$cut_points)[2]}'))
                          self$pretrained_agent <- pretrained_agent
                          assert_that(!is.null(algorithm), msg = 'Algorithm can\'t be NULL')
                          assert_that(algorithm %in% c('tdva', 'rlalgo', 'mcc'), msg = 'Only tdva, rlalgo and mcc supported')
                          self$algorithm <- algorithm
                        },
                      initialize_rl_framework = function(){
                        return(InitializeRLframework(self$features, self$algorithm))
                      })
)





CurPortfolio <- R6Class("CurPortfolio",
                      public = list(
                        assets           = NULL,
                        features         = NULL,
                        Nepisodes        = NULL,
                        cost             = NULL,
                        pretrained_agent = NULL,
                        algorithm        = NULL,
                        
                        initialize = function(assets, features, 
                                              Nepisodes, cost, pretrained_agent = NULL, 
                                              algorithm = NULL) {
                            
                          self$assets    <- assets
                          self$features  <- features
                          self$Nepisodes <- Nepisodes
                          self$cost      <- cost
                          assert_that(is.null(pretrained_agent) || dim(self$features)[2] == dim(pretrained_agent$cut_points)[2],
                                      msg = glue('pretrained agent has a different number of features {dim(self$features)[2]} than the number of features for state space {dim(pretrained_agent$cut_points)[2]}'))
                          self$pretrained_agent <- pretrained_agent
                          
                          # assert_that(!is.null(algorithm), msg = 'Algorithm can\'t be NULL')
                          # assert_that(algorithm %in% c('tdva', 'rlalgo', 'mcc'), msg = 'Only tdva, rlalgo and mcc supported')
                          self$algorithm <- algorithm
                          
                          
                        },
                        initialize_rl_framework = function(){
                          return(InitializeRLframework(self$features, self$algorithm))
                        },
                        train = function(algorithm = NULL){
                          assert_that(algorithm %in% c('tdva', 'rlalgo', 'mcc') ||
                                      !is.null(algorithm <- self$algorithm), 
                                      msg = 'Only tdva, rlalgo and mcc supported')
                          print(glue('Training with use of {algorithm} algorithm'))
                          if(algorithm == 'mcc'){
                            print('Former MCCbootstrap function')
                            return(MCCbootstrap(self))  
                          }else if(algorithm == 'rlalgo'){
                            return(Qcontrol(self))
                          }else{
                            return(Qcontrol(self))
                          }
                        })
)



