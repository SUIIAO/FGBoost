# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(haven)
library(plyr)
library(tidyr)
library(dplyr)

library(foreach)
library(doSNOW)

entry = read_dta('entry.dta')
entry = entry %>%
  select(caseid, b2_1, b2_2, b2_3, b2_4,
         b1, b3, b4, b5, b6, b7, e1, e8b, e11, e37) %>% 
  filter(!is.na(b2_1) & !is.na(b2_2) & !is.na(b2_3) & !is.na(b2_4) & !is.na(b1) & !is.na(b3) & !is.na(b4) & !is.na(b5) & !is.na(b6) & !is.na(b7) & !is.na(e1) & !is.na(e8b) & !is.na(e11) & !is.na(e37))

id = unique(entry$caseid)
n = length(id)

yM = entry %>%
  select(b2_1, b2_2, b2_3, b2_4) %>%
  mutate(b2_1 = sqrt(b2_1/100),
         b2_2 = sqrt(b2_2/100),
         b2_3 = sqrt(b2_3/100),
         b2_4 = sqrt(b2_4/100)) %>%
  as.matrix()

y = lapply(1:n, function(i){
  yM[i,]
})

x_pred = entry %>%
  select(b1, b3, b4, b5, b6, b7, e1, e8b, e11, e37)
names(x_pred) = c("life_satisfaction", "education", "marital_status", 
                  "num_children", "num_people", "income", "hours_per_week", 
                  "job_end", "weeks_searching", "credit_balance")

################################################################################
########################### 100 runs of 10-fold-CV #############################
################################################################################
ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)
text = "CV - r = %d is complete\n"
progress = function(r) {
  if(r%%1 == 0){
    cat(sprintf(text, r))
  }
}
opts = list(progress=progress)

Err = foreach (q = 1:100, .options.snow=opts) %:%
  foreach(k = 1:10) %dopar%  {
    library(plyr)
    library(tidyr)
    library(dplyr)
    library(foreach)
    library(stringr)
    library(haven)
    
    # GFR and LFR
    source("../../code/gsr-modify.R")
    source("../../code/lsr-modify.R")
    
    # FGBoost
    source("../../code/lcm.R")
    source("../../code/FGBoost.R")
    source("../../code/FGBoost_BuildTree.R")
    source("../../code/FGBoost_Prediction.R")
    
    # IFR
    source("../../Single-Index-Frechet/SIdxSpheReg.R")
    
    # SDR
    function_path = "../../DR4FrechetReg/Functions"
    function_sources = list.files(function_path,
                                   pattern="*.R$", full.names=TRUE,
                                   ignore.case=TRUE)
    sapply(function_sources, source, .GlobalEnv)
    

    n = nrow(yM)
    n_estimators = 50
    learning_rate = 0.1
    max_depth = 5
    min_samples_per_leaf = 100
    validation_fraction = 0.1
    
    # K-fold CV
    set.seed(q)
    cv_fold = data.frame(rd_ind = sample(1:nrow(x_pred),nrow(x_pred),replace = FALSE),
                         fold = rep(1:10,length.out=nrow(x_pred)))
    
    ind_test = cv_fold[cv_fold$fold==k,1]
    ind_remain = cv_fold[cv_fold$fold!=k,1]
    
    y_train = y[ind_remain]
    yM_train = do.call(rbind, y_train)
    y_test = y[ind_test]
    x_pred_train = x_pred[ind_remain,]
    x_pred_test = x_pred[ind_test,]
    
    x_pred_q = x_pred
    x_pred_train_mean = as.vector(colMeans(x_pred_train))
    x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
    
    x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd) ## standardize individually
    x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
    x_pred_q = t((t(x_pred_q) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
    
    
    ##############################################################################
    ################################# FGBoosting #################################
    ##############################################################################
    result = FGBoost(x_pred_train, y_train, n_estimators, learning_rate, max_depth,
                     optns = list(type = "compositional", impurity = "MSE", 
                                  min_samples_per_leaf = min_samples_per_leaf, 
                                  validation_fraction = validation_fraction,
                                  seed = q))
    n_estimator = which.min(result$err_valid)
    trees = result$trees[1:n_estimator]
    new_predictions = predict_boosted_model(trees = trees, F_0 = result$F_0, 
                                            X_new = x_pred_test, 
                                            learning_rate = learning_rate, 
                                            optns = list(type = "compositional"))
    err_FGBoost = sapply(1:nrow(x_pred_test), function(j){
      mse_loss(y_test[[j]], new_predictions[[j]], optns = list(type = "compositional"))
    })
    
    ##############################################################################
    #################################### GFR #####################################
    ##############################################################################
    res_glo = gsr(xin = as.matrix(x_pred_train), 
                  yin = as.matrix(yM_train),
                  xout = as.matrix(x_pred_test))
    err_GFR = sapply(1:nrow(x_pred_test), function(j){
      mse_loss(y_test[[j]], res_glo$yout[j,], optns = list(type = "compositional"))
    })
    
    ############################################################################
    ############################# SDR ##########################################
    ############################################################################
    dist.den = as.matrix(dist(yM_train, upper = T, diag = T))
    complexity = 1/2/(sum(dist.den[upper.tri(dist.den)])/choose(n,2))
    ygram = gram_matrix(yM_train, complexity = complexity, 
                        type="sphere", kernel="Laplacian")
    f = get("fopg")
    bhat = f(x=x_pred_train, y=ygram, d=1)$beta
    csd_train = as.matrix(x_pred_train)%*%bhat; csd_test = as.matrix(x_pred_test)%*%bhat;
    
    rg_SDR = range(csd_train)[2] - range(csd_train)[1]
    res_SDR = lsr(yin = yM_train, xin = csd_train, 
                  xout = csd_test,
                  optns = list(bw = 0.3*rg_SDR, kernel = "gauss"))
    err_SDR = sapply(1:nrow(x_pred_test), function(j){
      mse_loss(y_test[[j]], res_SDR$yout[j,], optns = list(type = "compositional"))
    })
    
    #############################################################################
    #################################### IFR ####################################
    #############################################################################
    res_sid = SIdxSpheReg(xin = as.matrix(x_pred_train),
                          yin = yM_train, iter = 100, bw = 1, M = 4)
    X_IFR_train = as.matrix(x_pred_train) %*% res_sid$est; X_IFR_test = as.matrix(x_pred_test) %*% res_sid$est;
    rg_IFR = apply(X_IFR_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
    res_IFR = lsr(yin = yM_train, xin = X_IFR_train, 
                  xout = X_IFR_test,
                  optns = list(bw = 0.3*rg_IFR, kernel = "gauss"))
    err_IFR = sapply(1:nrow(x_pred_test), function(j){
      mse_loss(y_test[[j]], res_IFR$yout[j,], optns = list(type = "compositional"))
    })

    k_err = data.frame(
      q = q, k = k, i = ind_test,
      FGBoost = err_FGBoost, GFR = err_GFR,
      SDR = err_SDR, IFR = err_IFR
    )
    
    return(k_err)
  }

stopCluster(cl)

library(stringr)
library(tidyverse)
library(knitr)

# Table 4
do.call(rbind, lapply(Err, function(x) do.call(rbind, x))) %>%
  select(q,k,FGBoost,GFR,SDR,IFR) %>%
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
    FGBoost_mean = mean(FGBoost),
    GFR_mean = mean(GFR),
    SDR_mean = mean(SDR),
    IFR_mean = mean(IFR)
  ) %>%
  mutate(FGBoost = sprintf("%.4f (%.4f)", FGBoost_mean, FGBoost_se),
         GFR = sprintf("%.4f (%.4f)", GFR_mean, GFR_se),
         SDR = sprintf("%.4f (%.4f)", SDR_mean, SDR_se),
         IFR = sprintf("%.4f (%.4f)", IFR_mean, IFR_se)
  ) %>%
  select(FGBoost, GFR, SDR, IFR) %>%
  knitr::kable(
    format = "latex",
    digits = 4)