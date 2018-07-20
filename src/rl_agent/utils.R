copyEnv <- function(from, to, names=ls(from, all.names=TRUE)) {
  mapply(assign, names, mget(names, from), list(to), 
         SIMPLIFY = FALSE, USE.NAMES = FALSE)
  invisible(NULL)
}


check_for_opt_args <- function(params, env){
  assert_that(!length(intersect(ls(envir = env), names(params))), 
              msg = glue("The following optional arguments are overwritting required arguments:
                         {paste(intersect(ls(), names(params)), collapse = ', ')}"))
  list2env(params, envir = env)
}

# ns_vars(env=env, values = T)
ns_vars <- function(env, values = T){
  walk(ls(envir = env),.f = function(x){ 
    x_tmp <- eval(parse(text = x), envir = env)
    print(glue('\n{x} [{class(x_tmp)}] '))
    if(!class(x_tmp) %in% c('function', 'closure', 'environment') && values) {
        print(head(x_tmp, 10))
        cat('\n')
    }
  })
}




