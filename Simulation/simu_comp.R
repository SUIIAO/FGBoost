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
    library(foreach)
    
    source("../code/FGBoost.R")
    source("../code/FGBoost_BuildTree.R")
    source("../code/FGBoost_Prediction.R")
    source("../code/lcm.R")
    
    # IFR
    source("../Single-Index-Frechet/SIdxSpheReg.R")
    source("../code/gsr-modify.R")
    source("../code/lsr-modify.R")
    
    # SDR
    function_path = "../DR4FrechetReg/Functions"
    function_sources = list.files(function_path,
                                   pattern="*.R$", full.names=TRUE,
                                   ignore.case=TRUE)
    sapply(function_sources, source, .GlobalEnv)
    
    # Frechet Forest
    library(RiemBase)
    library(spherepc)
    source("../Code_RFWLFR/FRFPackage2.R")
    source("../Code_RFWLFR/main.R")
    source("../Code_RFWLFR/IntrinsicMean_revise.R")
    
    # Setup
    set.seed(r)
    n_estimators = 100
    early_stopping_rounds = 10
    
    # Generate some sample data
    n_new = 100
    
    # Number of boosting rounds
    learning_rate = 0.05  # Learning rate
    max_depth = 3         # Maximum depth of the trees
    
    # Set Parameters
    X = matrix(c(runif(n, -1, 1), 
                 runif(n, -1, 1),
                 runif(n, 1, 2),
                 
                 rgamma(n, 3, 1),
                 rgamma(n, 4, 1),
                 rgamma(n, 5, 1),
                 
                 rbinom(n, 1, 0.1),
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
                     
                     rbinom(n_new, 1, 0.1),
                     rbinom(n_new, 1, 0.2),
                     rbinom(n_new, 1, 0.3),
                     rbinom(n_new, 1, 0.5)
    ), 
    n_new)
    
    y = lapply(1:n, function(i){
      a = 3*sin(pi*X[i,1])^2*X[i,7] + 3*cos(pi*X[i,2])^2*(1-X[i,7])
      b = -X[i,4]^(1/2)*X[i,8]+X[i,5]^(1/2)*(1-X[i,8])
      
      xx = b/(abs(a)+abs(b))
      phi <- pi * (xx + 2) / 8
      
      if(X[i,9] == 1){
        # 1
        temp <- c(cos(phi), sqrt(3) * sin(phi) / 2, sin(phi) / 2)
        # add noise
        e1 <- c(sin(phi), -sqrt(3) * cos(phi) / 2, -cos(phi) / 2)
        e2 <- c(0, 1 / 2, -sqrt(3) / 2)
      }else{
        # 2
        temp <- c(cos(phi), sin(phi) / 2, sqrt(3) * sin(phi) / 2)
        # add noise
        e1 <- c(sin(phi), -cos(phi) / 2, -sqrt(3) * cos(phi) / 2)
        e2 <- c(0, sqrt(3) / 2, -1/ 2)
      }

      U <- runif(1, min = -0.1, max = 0.1) * e1 + 
        runif(1, min = -0.1, max = 0.1) * e2
     
      cos(sqrt(sum(U^2))) * temp + sin(sqrt(sum(U^2))) * U / sqrt(sum(U^2))
    })
    
    y_true = lapply(1:n, function(i){
      a = 3*sin(pi*X[i,1])^2*X[i,7] + 3*cos(pi*X[i,2])^2*(1-X[i,7])
      b = -X[i,4]^(1/2)*X[i,8]+X[i,5]^(1/2)*(1-X[i,8])
      
      xx = b/(abs(a)+abs(b))
      phi <- pi * (xx + 2) / 8
      
      if(X[i,9] == 1){
        # 1
        temp <- c(cos(phi), sqrt(3) * sin(phi) / 2, sin(phi) / 2)
      }else{
        # 2
        temp <- c(cos(phi), sin(phi) / 2, sqrt(3) * sin(phi) / 2)
      }
      temp
    })
    y_new_true = lapply(1:n_new, function(i){
      a = 3*sin(pi*X_new[i,1])^2*X_new[i,7] + 3*cos(pi*X_new[i,2])^2*(1-X_new[i,7])
      b = -X_new[i,4]^(1/2)*X_new[i,8]+X_new[i,5]^(1/2)*(1-X_new[i,8])
      
      xx = b/(abs(a)+abs(b))
      phi <- pi * (xx + 2) / 8
      
      if(X_new[i,9] == 1){
        # 1
        temp <- c(cos(phi), sqrt(3) * sin(phi) / 2, sin(phi) / 2)
      }else{
        # 2
        temp <- c(cos(phi), sin(phi) / 2, sqrt(3) * sin(phi) / 2)
      }
      temp
    })
    
    # FGBoost
    result = FGBoost(X, y, n_estimators, learning_rate, max_depth,
                     optns = list(type = "compositional", impurity = "MSE",
                                  min_samples_per_leaf = 10, validation_fraction = 0.1,
                                  early_stopping_rounds = early_stopping_rounds))
    n_estimator = which.min(result$err_valid)
    trees = result$trees[1:n_estimator]
    new_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0,
                                            X_new = X_new,
                                            learning_rate = learning_rate,
                                            optns = list(type = "compositional"))
    test_err = mean(sapply(1:n_new, function(i){
      mse_loss(y_new_true[[i]], new_predictions[[i]], optns = list(type = "compositional"))
    }))
    
    # GFR
    yM = do.call(rbind, y)
    start = Sys.time()
    res_glo <- gsr(xin = as.matrix(X), 
                   yin = as.matrix(yM),
                   xout = as.matrix(X_new))

    test_glo = mean(sapply(1:n_new, function(i){
      mse_loss(y_new_true[[i]], res_glo$yout[i,], optns = list(type = "compositional"))
    }))
    
    # SDR
    X_SDR = apply(X,2,function(x) return((x-mean(x))/sd(x)))## standardize individually
    Xout_SDR = apply(X_new,2,function(x) return((x-mean(x))/sd(x)))## standardize individually
    
    dist.den = matrix(0, nrow = n, ncol = n)
    for(i in 1:(n-1)){
      for(j in (i+1):n){
        dist.den[i,j] = mse_loss(yM[i,], yM[j,], optns = list(type = "compositional"))
      }
    }
    dist.den = t(dist.den) + dist.den
    complexity = 1/2/(sum(dist.den[upper.tri(dist.den)]^2)/choose(n,2))
    ygram = gram_matrix(yM, complexity = complexity,
                        type="sphere", kernel="Laplacian")
    f = get("fopg")
    bhat = f(x=X_SDR, y=ygram, d=1)$beta
    csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
    
    rg_SDR = range(csd_train)
    res_SDR = lsr(yin = yM, xin = csd_train, 
                  xout = csd_test,
                  optns = list(bw = 0.2*(rg_SDR[2]-rg_SDR[1]), kernel = "gauss"))

    test_SDR = mean(sapply(1:n_new, function(i){
      mse_loss(y_new_true[[i]], res_SDR$yout[i,], optns = list(type = "compositional"))
    }))
    
    # # IFR
    # res_sid = SIdxSpheReg(xin = as.matrix(X), yin = yM, bw = 1.771488, M = 4, iter = 500)
    # X_SID_train = as.matrix(X) %*% res_sid$est; X_SID_test = as.matrix(X_new) %*% res_sid$est;
    # rg_SID = apply(X_SID_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    # res_SID = lsr(yin = do.call(rbind, y), xin = X_SID_train, xout = X_SID_test, 
    #               optns = list(bw = 0.2*rg_SID, kernel = "gauss"))
    # test_SID = mean(sapply(1:n_new, function(i){
    #   mse_loss(y_new_true[[i]], res_SID$yout[i,], optns = list(type = "compositional"))
    # }))
    
    # Random Forest Setup
    X_forest = list()
    X_forest$type = "scalar"
    X_forest$id = 1:n
    X_forest$X = X
    y_mat = do.call(rbind, y)
    q_mat = t(apply(y_mat, 1, sort))
    Y_forest = list()
    Y_forest$type = "sphere"
    Y_forest$id = 1:n
    Y_forest$Y = yM
    X_new_forest = list()
    X_new_forest$type = "scalar"
    X_new_forest$id = 1:n_new
    X_new_forest$X = X_new
    p = ncol(X)
    
    # FRF
    res_FRF = rfwlcfr(r = r, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=p, deep=max_depth, ntree=n_estimators, ncores=1)
    test_FRF = mean(sapply(1:n_new, function(i){
      mse_loss(y_new_true[[i]], res_FRF$res[i,], optns = list(type = "compositional"))
    }))
    
    # RFWLLFR
    res_rfwllfr = rfwllfr(r = r, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=p, deep=max_depth, ntree=n_estimators, ncores=1)
    test_rfwllfr = mean(sapply(1:n_new, function(i){
      mse_loss(y_new_true[[i]], res_rfwllfr$res[i,], optns = list(type = "compositional"))
    }))
    
    res = data.frame(
      n = n, r = r,
      FGBoost = test_err, GFR = test_glo, SDR = test_SDR, # IFR = test_SID, 
      FRF = test_FRF, RFWLLFR = test_rfwllfr)
    return(res)
  }

stopCluster(cl)


# Table 3
do.call(rbind, lapply(Err, function(x) do.call(rbind, x))) %>%
  dplyr::select(n, FGBoost, GFR, SDR, 
         # IFR, 
         FRF, RFWLLFR) %>%
  group_by(n) %>%
  dplyr::summarise(
    FGBoost_se = sd(FGBoost),
    GFR_se = sd(GFR),
    SDR_se = sd(SDR),
    # IFR_se = sd(IFR),
    FRF_se = sd(FRF),
    RFWLLFR_se = sd(RFWLLFR),
    FGBoost_mean = mean(FGBoost),
    GFR_mean = mean(GFR),
    SDR_mean = mean(SDR),
    # IFR_mean = mean(IFR),
    FRF_mean = mean(FRF),
    RFWLLFR_mean = mean(RFWLLFR)
  ) %>%
  mutate(FGBoost_mean = sprintf("%.4f", FGBoost_mean),
         FGBoost_se = sprintf("(%.4f)", FGBoost_se),
         GFR_mean = sprintf("%.4f", GFR_mean),
         GFR_se = sprintf("(%.4f)", GFR_se),
         SDR_mean = sprintf("%.4f", SDR_mean),
         SDR_se = sprintf("(%.4f)", SDR_se),
         # IFR_mean = sprintf("%.4f", IFR_mean),
         # IFR_se = sprintf("(%.4f)", IFR_se),
         FRF_mean = sprintf("%.4f", FRF_mean),
         FRF_se = sprintf("(%.4f)", FRF_se),
         RFWLLFR_mean = sprintf("%.4f", RFWLLFR_mean),
         RFWLLFR_se = sprintf("(%.4f)", RFWLLFR_se)
  ) %>%
  gather(key = method, value = Error, -c("n")) %>%
  group_by(n, method) %>%
  mutate(n_measure = sprintf("%d - %s", n, strsplit(method,"_")[[1]][2]),
         method = sprintf("%s", strsplit(method,"_")[[1]][1])) %>%
  spread(key = method, value = Error) %>%
  dplyr::select(n_measure, n, FGBoost, GFR, SDR, 
         # IFR, 
         FRF, RFWLLFR) %>%
  knitr::kable(
    format = "latex",
    digits = 4)
