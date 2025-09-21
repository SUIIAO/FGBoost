library(parallel)
library(glmnet)
library(stats)
library(Matrix)
library(osqp)
library(MLmetrics)
library(manifold) #for F Mean in sphere
library(trust) # for optimization (Newton Trust Region)
# library(frechet) # No sphere results

#### estimation performance calculator for estimates of the parameter, 
#### which are unit vectors
bias_calc2 <- function(est, true_beta, reps) {
  # true_beta: d vector
  # esti: reps*d matrix
  d <- length(true_beta)
  esti <- matrix(unlist(est), nrow = reps, byrow = T)
  extrinsic_mean <- colMeans(esti)
  
  if (sum(extrinsic_mean == 0) == d) {
    cat("Too few points\n")
    return(c(200, 200, 200))
  } else {
    intrinsic_mean <- as.vector(frechetMean(createM("Sphere"), t(esti)))
    bias <- sum(intrinsic_mean * true_beta)
    bias_angle <- acos(bias)
    dev <- mean(apply(esti, 1, function(x) sum(intrinsic_mean * x)))
    angle <- var(apply(esti, 1, function(x) acos(sum(intrinsic_mean * x))))
    #angle is dev?
    return(c(bias, bias_angle, dev, angle))
  }
}

l2norm <- function(x){
  sqrt(sum(x^2))
}

SpheGeoDist <- function(y1, y2){
  y1 = y1 / l2norm(y1)
  y2 = y2 / l2norm(y2)
  if(sum(y1 * y2) > 1){
    return(0)
  }else if(sum(y1 * y2) < -1){
    return(pi)
  }else{
    return(acos(sum(y1 * y2)))
  }
}

SpheGeoGrad <- function(y1, y2){
  y1 = y1 / l2norm(y1)
  y2 = y2 / l2norm(y2)
  tmp = 1 - sum(y1 * y2)^2
  if (tmp <= 0) {
    return(- Inf * y1)
  } else{
    return(-(tmp)^(-1/2) + y1)
  }
}

SpheGeoHess <- function(x,y) { #,tol = 1e-10){
  x = x / l2norm(x)
  y = y / l2norm(y)
  return(- sum(x * y) * (1 - sum(x * y) ^ 2) ^ (-1.5) * x %*% t(x))
}

# Kernels for local Fr reg
ker_gauss <- function(x) {
  return(exp(-x^2 / 2) / sqrt(2 * pi))
}

ker_unif <- function(x) {
  return(as.integer((x <= 1) & (x >= -1)))
}

ker_epan <- function(x, n = 1) {
  return((2 * n + 1) / (4 * n) * (1 - x^(2 * n)) * as.integer(abs(x) <= 1))
}

normalize <- function(x){
  x / sqrt(sum(x^2))
}

#### Local fr regression given any data, 
#### direction along which to compute projection, and bandw choice
LocLin <- function(dat, direc, xout, bw, ker = ker_gauss) {
  # dat: list of xin, yin
  ### xin: matrix of size d1*d2 (n * p)
  ### yin: matrix of size d1*m (n * m)
  # direc: directional vector, length d2 (p)
  # xout: t
  # bw: bandwidth b
  # ker: ker_gauss, ker_unif, ker_epan
  xin <- dat$xin
  yin <- dat$yin
  if (is.vector(xin)){
    stop("The number of observations is too small")
  }
  
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  if (d1 <3) {
    stop("The number of observations is too small")
  }
  projec <- xin %*% direc # d1-vector
  xin_eff <- projec
  yin_eff <- yin
  
  mu0 <- mean(ker((xin_eff - xout) / bw))
  mu1 <- mean(ker((xin_eff - xout) / bw) * (xin_eff - xout))
  mu2 <- mean(ker((xin_eff - xout) / bw) * ((xin_eff - xout)^2))
  s <- ker((xin_eff - xout) / bw) * (mu2 - mu1 * (xin_eff - xout)) / (mu0 * mu2 - mu1^2)
  s <- as.vector(s) # vector of length n
  m <- ncol(yin_eff)
  # Initial guess
  y0 <- colMeans(yin * s) 
  y0 <- y0 / l2norm(y0)
  
  # Check and adjust initial guess
  if (sum(sapply(1:d1, function(i) {sum(yin[i,] * y0)})[which(ker((xout - xin_eff) / bw) > 0)] > 1 - 1e-8) != 0) {
    y0 <- y0 + rnorm(m) * 1e-3
    y0 <- y0 / l2norm(y0)
  }
  
  # Compute value
  objfun = function(y){
    y <- y / l2norm(y)
    if(all(sapply(y, l2norm) == 1)){
      f <- Inf
    } else {
      f <- mean(s * sapply(1:d1, function(i) SpheGeoDist(yin[i,], y)^2))
    }
    g <- 2 * colMeans(t(
        sapply(1:d1, function(i) {
          SpheGeoDist(yin[i,], y) * SpheGeoGrad(yin[i,], y)})) * s)
    res = sapply(1:d1, function(i){
      grad_i = SpheGeoGrad(yin[i,], y)
      return((grad_i %*% t(grad_i) + SpheGeoDist(yin[i,], y) * SpheGeoHess(yin[i,], y)) * s[i])
    }, simplify = "array")
    h = 2 * apply(res, 1:2, mean)
    return(list(value=f, gradient=g, hessian=h))
    
    # res <- array(0, dim = c(m, m, d1))
    # for (i in 1:d1) {
    #   grad_i <- SpheGeoGrad(yin[i,], y)
    #   res[,,i] <- (outer(grad_i, grad_i) + 
    #                  SpheGeoDist(yin[i,], y) * SpheGeoHess(yin[i,], y)) * s[i]
    #   }
    #h <- 2 * apply(res, c(1,2), mean)
    #list(value = f, gradient = g, hessian = h)
  }
  
  # NewtonTrustRegion
  res <- trust(objfun, parinit = y0, rinit = 0.1, rmax = 1e5, minimize = T)
  return(res$argument / l2norm(res$argument))
}

### Bandwidth selection 5-fold CV function for the local Frechet regression along any projected direction
bwCV <- function(bw, dat, direc, ker = ker_gauss) {
  # dat: list of xin, yin
  ### xin: matrix of size d1*d2
  ### yin: matrix of size d1*m
  # direc: directional vector, length d2
  # bw: bandwidth b
  # ker: ker_gauss, ker_unif, ker_epan
  xin <- dat$xin
  yin <- dat$yin
  d1 <- nrow(xin)
  m = ncol(yin)
  projec <- xin %*% direc
  ind_cv <- split(1:d1, rep(1:5, length.out = d1))
  cv_err <- 0
  for (i in 1:5) {
    xin_eff <- xin[-ind_cv[[i]], ]
    yin_eff <- yin[-ind_cv[[i]], ]
    dat_cv <- list(xin = xin_eff, yin = yin_eff)
    
    for (k in 1:length(ind_cv[[i]])) {
      res <- LocLin(dat_cv, direc, projec[ind_cv[[i]][k]], bw, ker)
      cv_err <- cv_err + SpheGeoDist(res, yin[ind_cv[[i]][k], ])
    }
    cv_err <- cv_err / length(ind_cv[[i]])
  }
  return(cv_err / 5)
}

generate_data <- function(n, true_beta, link){
  d <- length(true_beta)
  xin <- matrix(runif(n * d), nrow = n)
  xin <- t(apply(xin, 1, normalize))
  link_proj = link(sapply(1:n, function(i){sum(true_beta * xin[i,])}))
  link_proj = (link_proj - min(link_proj))/(max(link_proj) - min(link_proj))
  err_sd = 0.2
  phi_true = acos(link_proj)
  theta_true = pi * link_proj
  m = 3
  ytrue = cbind(sin(phi_true) * cos(theta_true),
                sin(phi_true) * sin(theta_true),
                cos(phi_true))
  basis = list(
    b1 = cbind(cos(phi_true) * cos(theta_true),
               cos(phi_true) * sin(theta_true),
               -sin(phi_true)),
    b2 = cbind(sin(theta_true), -cos(theta_true), 0)
  )
  yin_tg = basis$b1 * rnorm(n, mean = 0, sd = err_sd) +
    basis$b2 * rnorm(n, mean = 0, sd = err_sd)
  yin = matrix(0, nrow = n, ncol = m)
  for(i in 1:n){
    tgNorm = sqrt(sum(yin_tg[i]^2))
    if(tgNorm < 1e-10){
      yin[i,] <- ytrue[i,]
    } else {
      yin[i,] <- sin(tgNorm) * yin_tg[i] / tgNorm + cos(tgNorm) * ytrue[i,]
    }
  }
  return(list(xin = xin, yin = yin))
}

#link <- list(function(x) x, function(x) x^2, function(x) exp(x))
#generate_data(10, true_beta = c(4,2,1,1), link = function(x){x^2})

### Binning step: given data and direction bins the support of the projection 
### and returns a representative point for the data (xin and yin)
#### Depends on the number of bins: M
binned_data <- function(dat, direc, M) {
  if (M < 4){
    stop("The number of binned data should be greater than 3.")
  }
  
  xin <- dat$xin
  yin <- dat$yin
  
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  m <- ncol(yin)
  
  if(d1 < M){
    stop("The number of binned data cannot exceed the number of observations.")
  }
  
  projec <- xin %*% direc
  range_of_projec <- seq(min(projec), max(projec), length.out = M)
  
  binned_xmean <- matrix(NA, M, d2)
  #binned_xmean[1, ] <- xin[which(projec <= range_of_projec[1]), ]#[1, ]
  binned_xmean[1, ] <- xin[which.min(projec), ]
  binned_ymean <- matrix(NA, M, m)
  binned_ymean[1, ] <- yin[which.min(projec), ]
  
  for (l in 2:(M - 1)) {
    binned_xmean[l, ] <- 
      xin[which((projec > range_of_projec[l]) & (projec <= range_of_projec[l + 1]))[1], ]
    binned_ymean[l, ] <- 
      yin[which((projec > range_of_projec[l]) & (projec <= range_of_projec[l + 1]))[1], ]
  }
  binned_xmean[M, ] <- xin[which.max(projec), ]
  binned_ymean[M, ] <- yin[which.max(projec), ]
  
  return(list(projec = projec, binned_xmean = binned_xmean, binned_ymean = binned_ymean))
}


### CV criterion to select M 
bwCV_M <- function(dat, direc, M, bw, ker = ker_gauss) {
  binned_dat <- binned_data(dat, direc, M)
  xin_binned <- binned_dat$binned_xmean
  yin_binned <- binned_dat$binned_ymean
  proj_binned <- xin_binned %*% direc
  d <- nrow(yin_binned)
  m <- ncol(yin_binned)
  cv_err <- 0
  
  for (i in 1:d) {
    xin_eff <- xin_binned[-i, ]
    yin_eff <- yin_binned[-i, ]
    dat_cv <- list(xin = xin_eff, yin = yin_eff)
    res <- LocLin(dat_cv, direc, proj_binned[i], bw, ker = ker_gauss)
    cv_err <- cv_err + SpheGeoDist(res, yin_binned[i, ])
  }
  return(cv_err / d)
}

# bw_min = maximum(maximum.([diff(xinSt), xinSt[2] .- minimum.(projec), 
#                            maximum.(projec) .- xinSt]))*1.1 / (ker == ker_gauss ? 3 : 1)
# bw_max = (maximum(projec) - minimum(projec))/3
# if bw_max < bw_min 
# if bw_min > bw_max*3/2
# #warning("Data is too sparse.")
# bw_max = bw_min*1.01
# else 
#   bw_max = bw_max*3/2
# end
# end
# #bw_range = [bw_min, bw_max]
# #bw_init = mean(bw_range)
# bw = optimize(x -> bwCV(dat, direc, x), bw_min, bw_max).minimizer
# end

#### Implements the selcetion of bandw for the local fr reg and, 
#### for the  optimal choice of bandw, selects the optimal bin size
tuning <- function(dat, direc, ker = ker_gauss){
  xin = dat$xin
  yin = dat$yin
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  projec = xin %*% direc
  xinSt = unique(sort(projec))
  # bw_min = max(max(c(diff(xinSt), xinSt[2] - min(projec), max(projec) - xinSt)))*1.1 / 
  #   ifelse(projec[ind_cv[[i]][k]], 3, 1)
  bw_min = max(c(diff(xinSt)))*1.1
  bw_max = (max(projec) - min(projec))/3
  if (bw_max < bw_min){
    if (bw_min > bw_max * 3/2){
      warning("Data is too sparse.")
      bw_max = bw_min * 1.01
    } else{
      bw_max = bw_max * 3/2
    }
  }
  #bw_range = c(bw_min, bw_max)
  #bw_init = mean(bw_range)
  bw = optim(par = runif(1, min = bw_min, max = bw_max), #runif initial value
             fn = bwCV, dat = dat, direc = direc,
             method = "Brent", #julia default
             lower = bw_min, upper = bw_max)$par
  M_range = ceiling(d1^(1/c(2:7)))
  M_range = unique(M_range[M_range > 3])
  #M_range = collect(range(minimum(binned_dat.binned_xmean),stop = maximum(binned_dat.binned_xmean), length = 30))
  if (length(M_range) >0){
    M_curr = M_range[1]
    cv_err_curr = bwCV_M(dat, direc, M_curr, bw)
    for (M in M_range){
      cv_err_test = bwCV_M(dat, direc, M, bw)
      if (cv_err_test < cv_err_curr){
        cv_err_curr <- cv_err_test
        M_curr <- M
      }
    }
    M = ceiling(M_curr)
  } else{
    M = d1
  }
  return(c(bw, M))
}

#### Main implementation function that returns 
#### the estimated direction parameter (unit vector) 
estimate_ichimura <- function(dat, bw = NULL, M = NULL, ker = ker_gauss) {
  xin <- dat$xin
  yin <- dat$yin
  
  if (is.vector(xin)){
    stop("The number of observations is too small")
  }
  
  if(!is.null(M)){
    if (M < 4){
      stop("The number of binned data should be greater than 3")
    }
  }
  
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  direc_curr_i <- normalize(rnorm(n = d2))
  
  if(is.null(bw) | is.null(M)){
    hh = tuning(dat, direc_curr_i)
    bw = hh[1]
    M = hh[2]
  }
  
  bw1 <- bw
  binned_dat <- binned_data(dat, direc_curr_i, M)
  d <- nrow(binned_dat$binned_xmean)
  proj_binned <- binned_dat$binned_xmean %*% direc_curr_i
  err <- 0
  
  for (l in 1:d) {
    res <- LocLin(dat, direc_curr_i, 
                  proj_binned[l],
                  bw, ker = ker_gauss)
    err <- err + SpheGeoDist(res, binned_dat$binned_ymean[l, ])
  }
  fdi_curr <- err / d
  
  for (i in 2:500) {
    direc_test <- normalize(rnorm(d2))
    binned_dat <- binned_data(dat, direc_test, M)
    proj_binned <- binned_dat$binned_xmean %*% direc_curr_i
    d <- nrow(binned_dat$binned_xmean)
    
    if (is.na(bw1)) {
      bw <- tuning(dat, direc_curr_i)[1]
    } # necessary??
    err <- 0
    
    for (l in 1:d) {
      res <- LocLin(dat, direc_test, proj_binned[l], bw, ker = ker_gauss)
      err <- err + SpheGeoDist(res, binned_dat$binned_ymean[l, ])
    }
    
    fdi_test <- err / d
    
    if (fdi_test < fdi_curr) {
      direc_curr_i <- direc_test
      fdi_curr <- fdi_test
    }
  }
  return(normalize(direc_curr_i))
}


# Function to estimate parameters with multiple replications
estimate_2pred <- function(n, reps = 100, bw, M, lk) {
  b <- c(4, 1.3, -2.5, 1.7)
  b0 <- normalize(b)
  d <- length(b0)
  
  est <- mclapply(
    1:reps, 
    FUN = function(x){
      dat <- generate_data(n, b0, lk)
      estimate_ichimura(dat)
    }, mc.cores = 2) ## should be changed for user convenience
  est_signed <- est
  try(est_signed <- lapply(est, function(x) x * sign(sum(b0 * x))))
  return(list(
    bias_calc2(est, b0, reps),
    bias_calc2(est_signed, b0, reps)
  ))
}

## More codes for rmpe

link <- list(function(x) x, function(x) x^2, function(x) exp(x))

for (ll in 1:3) {
  ddd <- estimate_2pred(n = 15, reps = 2, lk = link[[ll]])
  print(ddd)
}
