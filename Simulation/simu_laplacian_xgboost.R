# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(doSNOW)
library(foreach)
library(tidyverse)

# Set up parallel processing
ncores = 5
cl = makeCluster(ncores)
registerDoSNOW(cl)
text = "simulation1 - r = %d is complete\n"
progress = function(r) cat(sprintf(text, r))
opts = list(progress=progress)

output = foreach(r = 1:500, .options.snow=opts) %:%
  foreach(n = c(100, 200, 500, 1000)) %dopar% {
    library(stringr)
    library(doSNOW)
    library(xgboost)
    
    source("../XGBoost/xgboost.R")
    
    # Setup
    set.seed(r)
    n_new = 100
    m = 10
    
    # Set Parameters
    X = matrix(c(runif(n, -1, 1), 
                 runif(n, -1, 1),
                 runif(n, 1, 2),
                 
                 rgamma(n, 3, 1),
                 rgamma(n, 4, 1),
                 rgamma(n, 5, 1),
                 
                 rbinom(n, 1, 0.2),
                 rbinom(n, 1, 0.3),
                 rbinom(n, 1, 0.5)
    ), 
    n) 
    X_new = matrix(c(runif(n_new, -1, 1), 
                     runif(n_new, -1, 1),
                     runif(n_new, 1, 2),
                     
                     rgamma(n_new, 3, 1),
                     rgamma(n_new, 4, 1),
                     rgamma(n_new, 5, 1),
                     
                     rbinom(n_new, 1, 0.2),
                     rbinom(n_new, 1, 0.3),
                     rbinom(n_new, 1, 0.5)
    ), 
    n_new)
    
    y = lapply(1:n, function(i){
      a = 2 * sin(pi*X[i,1])^2*X[i,7] + cos(pi*X[i,2])^2*(1-X[i,7])
      b = X[i,4]*X[i,8]+X[i,5]*(1-X[i,8])
      
      Vec = -rbeta(m*(m-1)/2, shape1 = a, shape2 = b)
      temp = matrix(0, nrow = m, ncol = m)
      temp[lower.tri(temp)] = Vec
      temp <- temp + t(temp)
      diag(temp) = -colSums(temp)
      return(temp)
    })
    
    y_true = lapply(1:n, function(i){
      a = 2 * sin(pi*X[i,1])^2*X[i,7] + cos(pi*X[i,2])^2*(1-X[i,7])
      b = X[i,4]*X[i,8]+X[i,5]*(1-X[i,8])
      
      Vec = -rep(a/(a+b), m*(m-1)/2)
      temp = matrix(0, nrow = m, ncol = m)
      temp[lower.tri(temp)] = Vec
      temp <- temp + t(temp)
      diag(temp) = -colSums(temp)
      return(temp)
    })
    y_new_true = lapply(1:n_new, function(i){
      a = 2 * sin(pi*X_new[i,1])^2*X_new[i,7] + cos(pi*X_new[i,2])^2*(1-X_new[i,7])
      b = X_new[i,4]*X_new[i,8]+X_new[i,5]*(1-X_new[i,8])
      
      Vec = -rep(a/(a+b), m*(m-1)/2)
      temp = matrix(0, nrow = m, ncol = m)
      temp[lower.tri(temp)] = Vec
      temp <- temp + t(temp)
      diag(temp) = -colSums(temp)
      return(temp)
    })
    
    # XGBoost
    y_upper = lapply(1:n, function(j){
      y[[j]][upper.tri(y[[j]])]
    })
    y_upper = do.call(rbind, y_upper)
    
    # Train models
    models <- train_xgboost_multidim(
      X = X, 
      y_upper = y_upper, 
      n_estimators = 100,
      learning_rate = 0.05,
      max_depth = 3,
      early_stopping_rounds = 10
    )
    
    new_predictions <- predict_xgboost_multidim(models, X_new)
    
    y_new_upper = lapply(1:n_new, function(j){
      mat = matrix(0, m, m)
      mat[upper.tri(mat)] = new_predictions[j,]
      mat = mat + t(mat)
      diag(mat) = -colSums(mat)
      return(mat)
    })
    test_err = mean(sapply(1:n_new, function(i){
      sum((y_new_true[[i]] - y_new_upper[[i]])^2)
    }))
    
    res = list(n = n, r = r, XGboost = test_err)
    return(res)
  }

StopCluster(cl)

# Table 5
matrix(unlist(output), ncol=3, byrow=TRUE) %>%
  data.frame() %>%
  rename(n = 1, r = 2, XGboost = 3) %>%
  dplyr::select(n, XGboost) %>%
  group_by(n) %>%
  dplyr::summarise(
    XGboost_se = sd(XGboost),
    XGboost_mean = mean(XGboost)
  ) %>%
  mutate(XGboost = sprintf("%.3f (%.3f)", XGboost_mean, XGboost_se)) %>%
  select(n, XGboost)
