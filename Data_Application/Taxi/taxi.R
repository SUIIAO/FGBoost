# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(plyr)
library(tidyverse)
library(dplyr)

library(foreach)
library(doSNOW)


# Load Data
x_pred = readRDS("taxi_predictor_2017_2018.RData")
y_nw = readRDS("taxi_laplacian_2017_2018.RData")
meanLapM_y = sum((Reduce("+", y_nw) / length(y_nw))^2)

################################################################################
########################### 100 runs of 5-fold-CV #############################
################################################################################
ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)
text = "LOOCV - r = %d is complete\n"
progress = function(r) {
  if(r%%10 == 0){
    cat(sprintf(text, r))
  }
}
opts = list(progress=progress)

Err = foreach (q = 1:100, .options.snow=opts) %:%
  foreach(k = 1:5) %dopar%  {
    library(plyr)
    library(tidyr)
    library(dplyr)
    library(foreach)
    library(stringr)
    source('../../Network-Regression-with-Graph-Laplacians/src/gnr2.R')
    source('../../Network-Regression-with-Graph-Laplacians/src/lnr.R')
    source("../../Network-Regression-with-Graph-Laplacians/src/kerFctn.R")
    
    source("../../code/lcm.R")
    source("../../code/FGBoost.R")
    source("../../code/FGBoost_BuildTree.R")
    source("../../code/FGBoost_Prediction.R")
    
    # SIR
    source("../../Single-Index-Frechet/SIdxNetReg.R")
    
    # SDR
    function_path = "../../DR4FrechetReg/Functions"
    function_sources = list.files(function_path,
                                  pattern="*.R$", full.names=TRUE,
                                  ignore.case=TRUE)
    sapply(function_sources, source, .GlobalEnv)
    
    # Frechet Forest
    source("../../Code_RFWLFR/shape_revise.R")
    source("../../Code_RFWLFR/FRFPackage2.R")
    source("../../Code_RFWLFR/main.R")
    
    # Setting
    n = length(y_nw)
    learning_rate = 0.1
    n_estimators = 100
    max_depth = 2
    min_samples_per_leaf = 10
    validation_fraction = 0.1
    early_stopping_rounds = 10
    
    # K-fold CV
    set.seed(q)
    cv_fold = data.frame(rd_ind = sample(1:nrow(x_pred),nrow(x_pred),replace = FALSE),
                         fold = rep(1:5,length.out=nrow(x_pred)))
    
    ind_test = cv_fold[cv_fold$fold==k,1]
    ind_remain = cv_fold[cv_fold$fold!=k,1]
    
    y_nw_train = y_nw[ind_remain]
    y_nw_test = y_nw[ind_test]
    x_pred_train = x_pred[ind_remain,]
    x_pred_test = x_pred[ind_test,]
    
    x_pred_q = x_pred
    x_pred_train_mean = as.vector(colMeans(x_pred_train))
    x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
    
    x_pred_train_mean[which(names(x_pred_train) %in% c("MTWT", "FS"))] = 0
    x_pred_train_sd[which(names(x_pred_train) %in% c("MTWT", "FS"))] = 1
    
    x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd) ## standardize individually
    x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
    x_pred_q = t((t(x_pred_q) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
    
    
    ##############################################################################
    ################################# FGBoosting #################################
    ##############################################################################
    result = FGBoost(x_pred_train, y_nw_train, n_estimators, learning_rate, max_depth,
                     optns = list(type = "laplacian", impurity = "MSE",
                                  min_samples_per_leaf = min_samples_per_leaf,
                                  validation_fraction = validation_fraction,
                                  early_stopping_rounds = early_stopping_rounds,
                                  seed = q))
    
    n_estimator = which.min(result$err_valid)
    
    trees = result$trees[1:n_estimator]
    new_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0,
                                            X_new = x_pred_test,
                                            learning_rate = learning_rate,
                                            optns = list(type = "laplacian"))
    
    err_FGBoost = sapply(1:length(new_predictions), function(j){
      sum((y_nw_test[[j]] - new_predictions[[j]])^2)
    })/meanLapM_y
    
    ##############################################################################
    #################################### GFR #####################################
    ##############################################################################
    res_glo = gnr(gl = y_nw_train, x = x_pred_train, xOut = x_pred_test)
    
    err_GFR = sapply(1:nrow(x_pred_test), function(j){
      sum((y_nw_test[[j]] - res_glo$predict[[j]])^2)
    })/meanLapM_y
    
    ##############################################################################
    ################################# SDR  #######################################
    ##############################################################################
    y_mat = do.call(rbind, lapply(y_nw_train, function(x) as.vector(x)))
    dist.den = as.matrix(dist(y_mat, upper = T, diag = T))
    complexity = 1/2/(sum(dist.den[upper.tri(dist.den)])/choose(n,2))
    ygram = gram_matrix(y_nw_train, complexity = complexity, type="spd", kernel="Gaussian")
    f = get("fopg")
    bhat = f(x=x_pred_train, y=ygram, d=2)$beta
    csd_train = as.matrix(x_pred_train)%*%bhat; csd_test = as.matrix(x_pred_test)%*%bhat;
    
    rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    res_SDR = lnr(gl = y_nw_train, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR*0.1))
    
    err_SDR = sapply(1:nrow(x_pred_test), function(j){
      sum((y_nw_test[[j]] - res_SDR$predict[[j]])^2)
    })/meanLapM_y
    
    ##############################################################################
    ##################################### IFR ####################################
    ##############################################################################
    m = dim(y_nw_train[[1]])[1]
    start = Sys.time()
    y_mat = array(0, c(m, m, length(y_nw_train)))
    for(j in 1:length(y_nw_train)){
      y_mat[,,j] = y_nw_train[[j]]
    }
    res_sid = SIdxNetReg(xin = as.matrix(x_pred_train), Min = y_mat, bw = 1.0937, M = 4, iter = 50)
    X_SID_train = as.matrix(x_pred_train) %*% res_sid$est; X_SID_test = as.matrix(x_pred_test) %*% res_sid$est;
    rg_SID = apply(X_SID_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    res_IFR = lnr(gl = y_nw_train, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = rg_SID*0.1))
    end = Sys.time()
    time_IFR = as.numeric(end - start, units = "mins")
    
    err_IFR = sapply(1:nrow(x_pred_test), function(j){
      sum((y_nw_test[[j]] - res_IFR$predict[[j]])^2)
    })/meanLapM_y
    
    ##############################################################################
    ################################ Random Forest ###############################
    ##############################################################################
    X_forest <- list()
    X_forest$type <- "scalar"
    X_forest$id <- 1:nrow(x_pred_train)
    X_forest$X <- x_pred_train
    
    m = dim(y_nw_train[[1]])[1]
    y_mat = array(0, c(m, m, nrow(x_pred_train)))
    for(j in 1:nrow(x_pred_train)){
      y_mat[,,j] = y_nw_train[[j]]
    }
    Y_forest <- list()
    Y_forest$type <- "laplacian"
    Y_forest$id <- 1:nrow(x_pred_train)
    Y_forest$Y <- y_mat
    
    X_new_forest <- list()
    X_new_forest$type <- "scalar"
    X_new_forest$id <- 1:nrow(x_pred_test)
    X_new_forest$X <- x_pred_test
    deep = 5
    p = ncol(x_pred_train)
    method = "Euclidean"
    
    ##############################################################################
    ##################################### FRF ####################################
    ##############################################################################
    start = Sys.time()
    res_FRF = rfwlcfr(r = q, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest,
                      mtry=round(p/6), deep=deep, ntree=n_estimators, ncores=1)
    end = Sys.time()
    time_FRF = as.numeric(end - start, units = "mins")
    
    err_FRF = sapply(1:nrow(x_pred_test), function(j){
      sum((y_nw_test[[j]] - res_FRF$res[,,j])^2)
    })/meanLapM_y
    
    ##############################################################################
    ################################### RFWLLFR ##################################
    ##############################################################################
    start = Sys.time()
    res_RFWLLFR = rfwllfr(r = q, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest,
                          mtry=p, deep=max_depth, ntree=n_estimators, ncores=1)
    end = Sys.time()
    time_RFWLLFR = as.numeric(end - start, units = "mins")
    
    err_RFWLLFR = sapply(1:nrow(x_pred_test), function(j){
      sum((y_nw_test[[j]] - res_RFWLLFR$res[,,j])^2)
    })/meanLapM_y
    
    ##############################################################################
    ################################### Output ###################################
    ##############################################################################
    k_err = data.frame(
      q = q, k = k, i = ind_test,
      FGBoost = err_FGBoost, GFR = err_GFR, SDR = err_SDR,
      IFR = err_IFR, FRF = err_FRF, RFWLLFR = err_RFWLLFR)
    
    return(k_err)
  }
stopCluster(cl)

# Table 2
do.call(rbind, lapply(Err, function(x) do.call(rbind, x))) %>%
  group_by(q, k) %>%
  dplyr::summarise_all(mean) %>%
  group_by(q) %>%
  dplyr::summarise_all(mean) %>%
  ungroup() %>%
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
  mutate(FGBoost_mean = sprintf("%.5f", FGBoost_mean),
         FGBoost_se = sprintf("(%.5f)", FGBoost_se),
         GFR_mean = sprintf("%.5f", GFR_mean),
         GFR_se = sprintf("(%.5f)", GFR_se),
         SDR_mean = sprintf("%.5f", SDR_mean),
         SDR_se = sprintf("(%.5f)", SDR_se),
         IFR_mean = sprintf("%.5f", IFR_mean),
         IFR_se = sprintf("(%.5f)", IFR_se),
         FRF_mean = sprintf("%.5f", FRF_mean),
         FRF_se = sprintf("(%.5f)", FRF_se),
         RFWLLFR_mean = sprintf("%.5f", RFWLLFR_mean),
         RFWLLFR_se = sprintf("(%.5f)", RFWLLFR_se)
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