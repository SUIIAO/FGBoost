library(Matrix)
library(osqp)

SIdxDenReg2 = function(xin, qin, bw=NULL, M=NULL, iter = 500, ker = ker_gauss, lower = -Inf, upper = Inf) {
  ## xin: d1 by d2 matrix of input (d1: number of inputs, d2: dimension of predictors)
  ## qin: d1 by m matrix of quantile function  (m: length of quantile)
  ## bw: bandwidth b
  ## M: size of binning
  ## ker: ker_gauss, ker_unif, ker_epan
  
  if (is.vector(xin)){
    stop("The number of observations is too small")
  }
  if (!is.matrix(xin)){
    stop("xin should be matrix.")
  }
  if(!is.matrix(qin)){
    stop("qin should be matrix.")
  }
  
  d1 <- nrow(xin) # n
  d2 <- ncol(xin) # p
  direc_curr_i <- normalize(rnorm(n = d2))
  
  if(is.null(bw) | is.null(M)){
    if(is.null(bw)){
      param = DenTuning(xin, qin, direc_curr_i)
      bw = param[1]
      if(!is.null(M)){
        if (M < 4){stop("The number of binned data should be greater than 3")}
      } else{
        M = param[2]
      }
    }
  }
  
  binned_dat <- DenBinned_data(xin, qin, direc_curr_i, M)
  d <- nrow(binned_dat$binned_xmean) # = M
  proj_binned <- binned_dat$binned_xmean %*% direc_curr_i
  err <- 0
  
  for (l in 1:d) {
    res <- DenLocLin(
      xin, qin, 
      ## Consider using 
      ## binned_dat$binned_xmean,  binned_dat$binned_qmean, 
      ## instead, to match the paper 
      ## (although the Julia codes also use pre-binned data)
      direc_curr_i, 
      proj_binned[l],
      bw, ker = ker_gauss, lower = -Inf, upper = Inf)
    err <- err + mean((sort(res) - sort(binned_dat$binned_qmean[l, ]))^2)
  }
  fdi_curr <- err / d
  
  for (i in 2:iter) {
    direc_test <- normalize(rnorm(d2))
    ## updated: should CV for each new direction
    if(is.null(bw) | is.null(M)){
      if(is.null(bw)){
        param = DenTuning(xin, qin, direc_curr_i)
        bw = param[1]
        if(!is.null(M)){
          if (M < 4){stop("The number of binned data should be greater than 3")}
        } else{
          M = param[2]
        }
      }
    }
    
    binned_dat <- DenBinned_data(xin, qin, direc_test, M)
    proj_binned <- binned_dat$binned_xmean %*% direc_test ## typo fixed: updated from direc_curr_i
    d <- nrow(binned_dat$binned_xmean)
    
    err <- 0
    for (l in 1:d) {
      res <- DenLocLin(
        xin, qin, 
        ## Consider using 
        ## binned_dat$binned_xmean,  binned_dat$binned_qmean, 
        ## instead, to match the paper 
        ## (although the Julia codes also use pre-binned data)
        direc_test, 
        proj_binned[l], bw, ker = ker_gauss, lower = -Inf, upper = Inf)
      err <- err + mean((sort(res) - sort(binned_dat$binned_qmean[l, ]))^2)
    }
    
    fdi_test <- err / d
    
    if (fdi_test < fdi_curr) {
      direc_curr_i <- direc_test
      fdi_curr <- fdi_test
    }
  }
  return(normalize(direc_curr_i))
}

## Notice: R and Julia give inconsistent results

