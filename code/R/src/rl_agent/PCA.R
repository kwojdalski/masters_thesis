install.packages("svd")
# library(svd)

coefficients <- function (df_attributes, threshold){
  # data standardization
  df_attributes <- scale(df_attributes)

  # Singular Value Decomposition
  s <- svd(df_attributes)
  # df_atributes = U * d*I * V', where I denotes identity matrix

  ### variance
  # the square of each singular value is proportional to the variance explained by each singular vector
  variance <- (s$d)^2
  # total variance
  all_variance <- sum(variance)
  # percetnage of variance explained
  var_explained <- cumsum(variance) / all_variance

  # number of chosen singular vectors to explain at least 'threshold' of total variance
  chosen <- sum(as.numeric( var_explained < threshold )) + 1

  # return PCA loadings -- rows of V matrix
  coeff <- s$v[1:chosen,]
  return( t(coeff) )
}


reduced_df_attributes<- function (df_attributes, coefficients){
  n <- ncol(coefficients)
  # return chosen Principal Components
  return( df_attributes[,1:n] %*% coefficients)
}
