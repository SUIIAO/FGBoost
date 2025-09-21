# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(doSNOW)
library(foreach)
library(tidyverse)

# Set up parallel processing
ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)
text = "simulation1 - r = %d is complete\n"
progress = function(r) cat(sprintf(text, r))
opts = list(progress=progress)

Err = foreach(r = 1:500, .options.snow=opts) %:%
  foreach(n = c(100, 200, 500, 1000)) %dopar% {
    library(truncnorm)
    library(frechet)
    library(stringr)
    library(doSNOW)
    
    source("../code/FGBoost.R")
    source("../code/FGBoost_BuildTree.R")
    source("../code/FGBoost_Prediction.R")
    source("../code/lcm.R")
    
    source("../Wasserstein-regression-with-empirical-measures-main/code/grem.R")
    source("../Wasserstein-regression-with-empirical-measures-main/code/lcm.R")
    source("../Wasserstein-regression-with-empirical-measures-main/code/lrem.R")
    source("../Wasserstein-regression-with-empirical-measures-main/code/bwCV.R")
    source("../Wasserstein-regression-with-empirical-measures-main/code/kerFctn.R")
    
    # IFR
    source("../Single-Index-Frechet/SIdxDenReg.R")
    
    # SDR
    function_path = "../DR4FrechetReg/Functions"
    function_sources = list.files(function_path,
                                  pattern="*.R$", full.names=TRUE,
                                  ignore.case=TRUE)
    sapply(function_sources, source, .GlobalEnv)
    
    # Frechet Forest
    source("../Code_RFWLFR/FRFPackage2.R")
    source("../Code_RFWLFR/main.R")
    
    # Setup
    set.seed(r)
    n_estimators = 100
    early_stopping_rounds = 10
    n_new = 100
    N = 100
    learning_rate = 0.05
    max_depth = 3
    
    # Set Parameters
    X = matrix(c(runif(n, 0, 1), 
                 runif(n, -1, 1),
                 rbinom(n, 1, 0.1),
                 rbinom(n, 1, 0.2),
                 rnorm(n, 0, 1),
                 rnorm(n, 0, 1) ,
                 runif(n,-2,2),
                 rbinom(n,1,0.5),
                 rnorm(n,0,1)
    ), 
    n) 
    X_new = matrix(c(runif(n_new, 0, 1),    
                     runif(n_new, -1, 1),   
                     rbinom(n_new, 1, 0.1), 
                     rbinom(n_new, 1, 0.2), 
                     rnorm(n_new, 0, 1),    
                     rnorm(n_new, 0, 1),    
                     runif(n_new,-2,2),     
                     rbinom(n_new,1,0.5),   
                     rnorm(n_new,0,1)       
    ), 
    n_new) 
    
    y = lapply(1:n, function(i){
      expect_eta_Z = sin(pi*X[i,1]) - X[i,3] * X[i,5]^2/2
      mu = rnorm(1, mean = expect_eta_Z, sd = 0.5)
      
      expect_sigma_Z = 1 + 2*cos(pi/2*X[i,2]) + X[i,4] * X[i,6]^2/2
      sigma = rgamma(1, shape = expect_sigma_Z^2, scale = 1/expect_sigma_Z)
      
      sort(
        c(-2, rtruncnorm(N, a = -2, b = 2, mean = mu, sd = sigma), 2)
      )
    })
    
    y_true = lapply(1:n, function(i){
      qtruncnorm(0:(N+1)/(N+1), 
                 a = -2, 
                 b = 2,
                 mean = sin(pi*X[i,1]) - X[i,3] * X[i,5]^2/2,
                 sd = 1 + 2*cos(pi/2*X[i,2]) + X[i,4] * X[i,6]^2/2
      )
    })
    y_new_true = lapply(1:n_new, function(i){
      qtruncnorm(0:(N+1)/(N+1),
                 a = -2, 
                 b = 2,
                 mean = sin(pi*X_new[i,1]) - X_new[i,3] * X_new[i, 5]^2/2,
                 sd = 1 + 2*cos(pi/2*X_new[i,2]) + X_new[i,4] * X_new[i,6]^2/2
      )
    })
    
    # FGBoost
    result = FGBoost(X, y, n_estimators, learning_rate, max_depth,
                     optns = list(type = "measure", impurity = "MSE",
                                  min_samples_per_leaf = 10, validation_fraction = 0.1, 
                                  early_stopping_rounds = early_stopping_rounds))
    n_estimator = which.min(result$err_valid)
    trees = result$trees[1:n_estimator]
    final_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0,
                                              X_new = X,
                                              learning_rate = learning_rate,
                                              optns = list(type = "measure"))
    new_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0,
                                            X_new = X_new,
                                            learning_rate = learning_rate,
                                            optns = list(type = "measure"))
    test_err = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - new_predictions[[i]])^2)
    }))
    
    # GFR
    yM = matrix(unlist(y), nrow = n, byrow = TRUE)
    res_glo = grem(y = y, x = X, xOut = X_new, optns = list(lower = -2, upper = 2))
    test_glo = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - res_glo$qp[i,])^2)
    }))
    
    # SDR
    X_SDR = apply(X,2,function(x) return((x-mean(x))/sd(x)))## standardize individually
    Xout_SDR = apply(X_new,2,function(x) return((x-mean(x))/sd(x)))## standardize individually
    
    y_mat = do.call(rbind, y)
    dist.den = as.matrix(dist(y_mat, upper = T, diag = T))/sqrt(N)
    complexity = 1/2/(sum(dist.den[upper.tri(dist.den)]^2)/choose(n,2))
    ygram = gram_matrix(y_mat, complexity = complexity,
                        type="distribution", kernel="Gaussian")
    f = get("fopg")
    bhat = f(x=X_SDR, y=ygram, d=2)$beta
    csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
    
    rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    res_SDR = lrem(y = y, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR*0.1, lower = -2, upper = 2))
    
    test_SDR = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - res_SDR$qp[i,])^2)
    }))
    
    # IFR
    q_mat = t(apply(y_mat, 1, sort))
    res_sid = SIdxDenReg(xin = as.matrix(X), qin = q_mat)
    X_SID_train = as.matrix(X) %*% res_sid$est; X_SID_test = as.matrix(X_new) %*% res_sid$est
    res_SID = lrem(y = y, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = res_sid$bw, lower = -2, upper = 2))
    
    test_IFR = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - res_SID$qp[i,])^2)
    }))
    
    # Random Forest
    X_forest = list()
    X_forest$type = "scalar"
    X_forest$id = 1:n
    X_forest$X = X
    y_mat = do.call(rbind, y)
    q_mat = t(apply(y_mat, 1, sort))
    Y_forest = list()
    Y_forest$type <- "distribution"
    Y_forest$id = 1:n
    Y_forest$Y = q_mat
    X_new_forest = list()
    X_new_forest$type = "scalar"
    X_new_forest$id = 1:n_new
    X_new_forest$X = X_new
    p = ncol(X)
    nqSup = ncol(q_mat) 
    qSup = seq(0,1,length.out = nqSup)
    
    # FRF
    res_FRF = rfwlcfr(r = r, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=ceiling(4/5*p), deep=2, ntree=n_estimators, ncores=1)
    test_FRF = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - res_FRF$res[i,])^2)
    }))
    
    # RFWLLFR
    res_rfwllfr = rfwllfr(r = r, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=ceiling(4/5*p), deep=5, ntree=n_estimators, ncores=1)
    test_rfwllfr = mean(sapply(1:n_new, function(i){
      mean((y_new_true[[i]] - res_rfwllfr$res[i,])^2)
    }))
    
    res = data.frame(n = n, r = r,
                     FGBoost = test_err, GFR = test_glo,
                     SDR = test_SDR, IFR = test_IFR,
                     FRF = test_FRF, RFWLLFR = test_rfwllfr)
    return(res)
  }

stopCluster(cl)

# Table 1
do.call(rbind, lapply(Err, function(x) do.call(rbind, x))) %>%
  select(n, FGBoost, GFR, SDR, IFR, FRF, RFWLLFR) %>%
  group_by(n) %>%
  dplyr::summarise(
    FGBoost_se = sd(FGBoost),
    GFR_se = sd(GFR),
    SDR_se = sd(SDR),
    IFR_se = sd(IFR),
    FRF_se = sd(FRF),
    RFWLLFR_se = sd(RFWLLFR),
    FGBoost_mean = mean(FGBoost),
    GFR_mean = mean(GFR),
    SDR_mean = mean(SDR),
    IFR_mean = mean(IFR),
    FRF_mean = mean(FRF),
    RFWLLFR_mean = mean(RFWLLFR)
  ) %>%
  mutate(FGBoost_mean = sprintf("%.3f", FGBoost_mean),
         FGBoost_se = sprintf("(%.3f)", FGBoost_se),
         GFR_mean = sprintf("%.3f", GFR_mean),
         GFR_se = sprintf("(%.3f)", GFR_se),
         SDR_mean = sprintf("%.3f", SDR_mean),
         SDR_se = sprintf("(%.3f)", SDR_se),
         IFR_mean = sprintf("%.3f", IFR_mean),
         IFR_se = sprintf("(%.3f)", IFR_se),
         FRF_mean = sprintf("%.3f", FRF_mean),
         FRF_se = sprintf("(%.3f)", FRF_se),
         RFWLLFR_mean = sprintf("%.3f", RFWLLFR_mean),
         RFWLLFR_se = sprintf("(%.3f)", RFWLLFR_se)
  ) %>%
  gather(key = method, value = Error, -c("n")) %>%
  group_by(n, method) %>%
  mutate(n_measure = sprintf("%d - %s", n, strsplit(method,"_")[[1]][2]),
         method = sprintf("%s", strsplit(method,"_")[[1]][1])) %>%
  spread(key = method, value = Error) %>%
  select(n_measure, n, FGBoost, GFR, SDR, IFR, FRF, RFWLLFR) %>%
  knitr::kable(
    format = "latex",
    digits = 4)




