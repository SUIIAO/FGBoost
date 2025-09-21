# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(frechet)
library(pracma)
library(dplyr)
library(frechet)
library(fdadensity)
library(tidyverse)

library(foreach)
library(doSNOW)
library(reshape2)

# Load the mortality data
mortality = readRDS("mortality.RData")

# Extract predictor matrix and density data
x_pred = mortality$pred

# Convert density data to quantile functions
quan = foreach(i = (1:nrow(x_pred)), .combine = "rbind") %do% {
  x = mortality$density[[i]]$x
  y = mortality$density[[i]]$y
  y.quantile = dens2quantile(dens = y, dSup = x)
}
n = nrow(x_pred)

ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)
text = "LOOCV - r = %d is complete\n"
progress = function(r) {
  if(r%%10 == 0){
    cat(sprintf(text, r))
  }
}
opts = list(progress=progress, verbose=FALSE)

Err = foreach(q = 1:n, .options.snow=opts) %dopar% { 
  library(frechet)
  library(pracma)
  library(dplyr)
  library(frechet)
  library(fdadensity)
  library(tidyverse)
  
  library(foreach)
  library(doSNOW)
  library(reshape2)
  
  # FGBoost
  source("../../code/lcm.R")
  source("../../code/FGBoost.R")
  source("../../code/FGBoost_BuildTree.R")
  source("../../code/FGBoost_Prediction.R")
  
  # GFR and LFR
  source("../../Wasserstein-regression-with-empirical-measures-main/code/grem.R")
  source("../../Wasserstein-regression-with-empirical-measures-main/code/lcm.R")
  source("../../Wasserstein-regression-with-empirical-measures-main/code/lrem.R")
  source("../../Wasserstein-regression-with-empirical-measures-main/code/bwCV.R")
  source("../../Wasserstein-regression-with-empirical-measures-main/code/kerFctn.R")
  
  # IFR
  source("../../Single-Index-Frechet/SIdxDenReg.R")
  
  # SDR
  function_path = "../../DR4FrechetReg/Functions"
  function_sources = list.files(function_path,
                                 pattern="*.R$", full.names=TRUE,
                                 ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  gram_wass = function(dist.den, complexity){
    n = dim(dist.den)[1]
    kupper = dist.den[upper.tri(dist.den,diag = FALSE)]
    k = matrix(0, nrow = n, ncol = n)
    k[upper.tri(k,diag = FALSE)]=kupper^2
    k = k+t(k)
    sigma2 = sum(k)/choose(n,2)
    gamma = complexity/(sigma2)
    print(gamma)
    return(exp(-gamma*k))
  }
  
  # Random Forest
  source("../../Code_RFWLFR/FRFPackage2.R")
  source("../../Code_RFWLFR/main.R")
  
  ind_test = q
  ind_remain = setdiff(1:162, ind_test)
  
  quan_train = quan[ind_remain,]
  quan_test = matrix(quan[ind_test,], nrow = length(ind_test))
  x_pred_train = x_pred[ind_remain,]
  x_pred_test = matrix(x_pred[ind_test,], nrow = length(ind_test))
  
  x_pred_train_mean = as.vector(colMeans(x_pred_train))
  x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
  x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd) ## standardize individually
  x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
  
  learning_rate = 0.1
  n_estimators = 50
  max_depth = 2
  min_samples_per_leaf = 10
  validation_fraction = 0.1
  early_stopping_rounds = 10
  
  y = lapply(1:nrow(quan_train), function(j) quan_train[j,])
  
  ##############################################################################
  ############################## FGBoosting ####################################
  ##############################################################################
  result = FGBoost(x_pred_train, y, n_estimators = n_estimators, 
                   learning_rate = learning_rate, max_depth=max_depth,
                   optns = list(type = "measure", impurity = "MSE", 
                                min_samples_per_leaf = min_samples_per_leaf,
                                early_stopping_rounds = early_stopping_rounds,
                                validation_fraction = validation_fraction, seed = q)) 
  
  n_estimator = which.min(result$err_valid)
  trees = result$trees[1:n_estimator]
  new_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0, 
                                          X_new = x_pred_test, 
                                          learning_rate = learning_rate, 
                                          optns = list(type = "measure"))
  err_FGBoost = sapply(1:length(ind_test), function(i) mean((quan_test[i,] - new_predictions[[i]])^2))
  
  ##############################################################################
  ############################ SDR Zhang Qi ####################################
  ##############################################################################
  X_SDR = x_pred_train ## standardize individually
  Xout_SDR = x_pred_test ## standardize individually
  dist.den_cv = as.matrix(dist(quan_train, upper = TRUE, diag = TRUE))/sqrt(ncol(quan_train))
  complexity = 15 # 1/2/(sum(dist.den[upper.tri(dist.den)]^2)/choose(n,2))
  y_gram = gram_wass(dist.den=dist.den_cv, complexity=complexity)
  f = get("fopg")
  bhat = f(x=X_SDR, y=y_gram, d=2)$beta
  csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
  
  rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_SDR = lrem(y = y, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR*0.1))
  err_sdr = sapply(1:length(ind_test), function(j) mean((res_SDR$qp[j,] - quan_test[j,])^2))
  
  ##############################################################################
  ######################## Global Network Regression ###########################
  ##############################################################################
  res_gn = grem(y = y, x = x_pred_train, xOut = x_pred_test)
  err_gn = sapply(1:nrow(quan_test), function(j) mean((res_gn$qp[j,] - quan_test[j,])^2))
  
  ##############################################################################
  ############################### Single Index Model ###########################
  ##############################################################################
  res_sid = SIdxDenReg(xin = as.matrix(x_pred_train), qin = quan_train)
  X_SID_train = as.matrix(x_pred_train) %*% res_sid$est; X_SID_test = as.matrix(x_pred_test) %*% res_sid$est;
  res_SID = lrem(y = y, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = res_sid$bw, lower = 0, upper = 100))
  err_IFR = sapply(1:nrow(quan_test), function(j) mean((res_SID$qp[j,] - quan_test[j,])^2))
  
  ##############################################################################
  ######################## Frechet Random Forest ###############################
  ##############################################################################
  #RFWLCFE
  X_forest <- list()
  X_forest$type <- "scalar"
  X_forest$id <- 1:nrow(x_pred_train)
  X_forest$X <- x_pred_train
  
  y_mat = do.call(rbind, y)
  q_mat = t(apply(y_mat, 1, sort))
  Y_forest = list()
  Y_forest$type = "distribution"
  Y_forest$id = 1:nrow(q_mat)
  Y_forest$Y = q_mat
  
  X_new_forest = list()
  X_new_forest$type = "scalar"
  X_new_forest$id = 1:length(ind_test)
  X_new_forest$X = x_pred_test
  deep = 10
  p = ncol(x_pred_train)
  nqSup = ncol(q_mat) 
  qSup = seq(0,1,length.out = nqSup)
  
  ##############################################################################
  ################################# FRF ########################################
  ##############################################################################
  res_FRF = rfwlcfr(r = q, Scalar=X_forest, Y=Y_forest, 
                    Xout=X_new_forest, 
                    mtry=ceiling(1/5*p), 
                    deep=deep, ntree=100, ncores=1)
  err_FRF = sapply(1:nrow(x_pred_test), function(j) mean((quan_test[j,] - res_FRF$res[j,])^2))
  
  ##############################################################################
  ############################## RFWLLFR #######################################
  ##############################################################################
  res_RFWLLFR = rfwllfr(r = q, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, 
                        mtry=ceiling(1/5*p), 
                        deep=deep, ntree=100, ncores=1)
  err_RFWLLFR = sapply(1:nrow(x_pred_test), function(j) mean((quan_test[j,] - res_RFWLLFR$res[j,])^2))

  ##############################################################################
  ############################## Save Results ##################################
  ##############################################################################
  res = data.frame(q = q, 
                   FGBoost = err_FGBoost, GFR = err_gn, SDR = err_sdr,  
                   IFR = err_IFR,  FRF = err_FRF, RFWLLFR = err_RFWLLFR)
  
  return(res)
}

stopCluster(cl)

# Table 2
do.call(rbind, Err) %>%
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
  mutate(FGBoost_mean = sprintf("%.2f", FGBoost_mean),
         FGBoost_se = sprintf("(%.2f)", FGBoost_se),
         GFR_mean = sprintf("%.2f", GFR_mean),
         GFR_se = sprintf("(%.2f)", GFR_se),
         SDR_mean = sprintf("%.2f", SDR_mean),
         SDR_se = sprintf("(%.2f)", SDR_se),
         IFR_mean = sprintf("%.2f", IFR_mean),
         IFR_se = sprintf("(%.2f)", IFR_se),
         FRF_mean = sprintf("%.2f", FRF_mean),
         FRF_se = sprintf("(%.2f)", FRF_se),
         RFWLLFR_mean = sprintf("%.2f", RFWLLFR_mean),
         RFWLLFR_se = sprintf("(%.2f)", RFWLLFR_se)
  ) %>%
  gather(key = method, value = Error) %>%
  group_by(method) %>%
  mutate(n_measure = sprintf("%s",strsplit(method,"_")[[1]][2]),
         method = sprintf("%s", strsplit(method,"_")[[1]][1])) %>%
  spread(key = method, value = Error) %>%
  select(n_measure, FGBoost, GFR, SDR, IFR, FRF, RFWLLFR) %>%
  knitr::kable(
    format = "latex",
    digits = 4)
