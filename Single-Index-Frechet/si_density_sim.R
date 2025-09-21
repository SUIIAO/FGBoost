library(parallel)
library(glmnet)
library(stats)
library(Matrix)
library(osqp)
library(MLmetrics)
library(manifold) #for F Mean in sphere
# library(frechet) #for kernels

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

#### Local fr regression given any data, 
#### direction along which to compute projection, and bandw choice
LocLin <- function(dat, direc, xout, bw, ker = ker_gauss, 
                   lower = -Inf, upper = Inf) {
  # dat: list of xin, qin
  ### xin: matrix of size d1*d2
  ### qin: matrix of size d1*101 (look at rbeta part)
  # direc: directional vector, length d2
  # xout: t
  # bw: bandwidth b
  # ker: ker_gauss, ker_unif, ker_epan
  xin <- dat$xin
  qin <- dat$qin
  if (is.vector(xin)){
    stop("The number of observations is too small")
  }
  
  
  
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  
  if (d1 <3) {
    stop("The number of observations is too small")
  }
  projec <- xin %*% direc
  xin_eff <- projec
  qin_eff <- qin

  
  mu0 <- mean(ker((xin_eff - xout) / bw))
  mu1 <- mean(ker((xin_eff - xout) / bw) * (xin_eff - xout))
  mu2 <- mean(ker((xin_eff - xout) / bw) * ((xin_eff - xout)^2))
  s <- ker((xin_eff - xout) / bw) * (mu2 - mu1 * (xin_eff - xout)) / (mu0 * mu2 - mu1^2)
  s <- as.vector(s)
  m <- ncol(qin_eff)
  b0 <- c(lower, rep(0, m - 1), -upper)
  
  Pmat <- Diagonal(n = m)
  Amat <- bandSparse(m+1, m, k = c(0, -1), diag = list(rep(1, m), rep(-1, m)))
  
  gx <- colMeans(qin_eff * s)
  
  prob <- osqp(Pmat, -gx, Amat, b0, pars = osqpSettings(verbose = FALSE))
  results <- prob$Solve()$x
  return(results)
}

#### Bandwidth selection 5-fold CV function for the local Frechet regression 
#### along any projected direction
bwCV <- function(bw, dat, direc, ker = ker_gauss, lower = -Inf, upper = Inf) {
  # dat: list of xin, qin
  ### xin: matrix of size d1*d2
  ### qin: matrix of size d1*101 (look at rbeta part)
  # direc: directional vector, length d2
  # bw: bandwidth b
  # ker: ker_gauss, ker_unif, ker_epan
  xin <- dat$xin
  qin <- dat$qin
  
  if (is.vector(xin)){
    stop("The number of observations is too small")
  } 
  
  d1 <- nrow(xin)
  d2 <- ncol(xin)
  m <- ncol(qin)

  
  projec <- xin %*% direc
  ind_cv <- split(1:d1, rep(1:5, length.out = d1))
  cv_err <- 0
  
  for (i in 1:5) {
    xin_eff <- xin[-ind_cv[[i]], ]
    qin_eff <- qin[-ind_cv[[i]], ]
    dat_cv <- list(xin = xin_eff, qin = qin_eff)
    
    for (k in 1:length(ind_cv[[i]])) {
      res <- LocLin(dat_cv, direc, projec[ind_cv[[i]][k]], bw, ker, lower = -Inf, upper = Inf)
      cv_err <- cv_err + mean((sort(res) - sort(qin[ind_cv[[i]][k], ]))^2)
    }
    cv_err <- cv_err / length(ind_cv[[i]])
  }
  return(cv_err / 5)
}

# Data generation mechanisms
generate_data <- function(n, true_beta, link) {
  d <- length(true_beta)
  xin <- matrix(rnorm(n * d), n, d)
  qSup <- qbeta(seq(0.01, 0.99, by = 0.01), 1/2, 1/2)
  m <- length(qSup) + 2
  qin <- matrix(0, n, m)
  for (i in 1:n) {
    qin[i, ] <- qnorm(c(1e-6, qSup, 1 - 1e-6), 
                      mean = link(sum(true_beta * xin[i, ])), sd = 0.1)
    #doesn't match the paper's setting
  }
  return(list(xin = xin, qin = qin))
}

# Generate data with alternative mechanism
generate_data2 <- function(n, true_beta, link) {
  d <- length(true_beta)
  xin <- matrix(rnorm(n * d), n, d)
  qSup <- qbeta(seq(0.01, 0.99, by = 0.01), 1/2, 1/2)
  m <- length(qSup) + 2
  qin <- matrix(0, n, m)
  for (i in 1:n) {
    k <- sample(c(-2, -1, 1, 2), 1) #no \pm 3, no -1
    Tk <- function(x) {x - (sin(k * x) / abs(k))}
    qin[i, ] <- Tk(qnorm(c(1e-6, qSup, 1 - 1e-6), 
                         mean = link(sum(true_beta * xin[i, ])), sd = 0.1))
  }
  return(list(xin = xin, qin = qin))
}

# # Given a multivariate predictor and a given direction, calculates the projection
# generate_proj <- function(xin, direc) {
#   projec <- xin %*% direc
#   return(projec)
# }

### Binning step: given data and direction bins the support of the projection 
### and returns a representative point for the data (xin and qin)
#### Depends on the number of bins: M
binned_data <- function(dat, direc, M) {
  
  if (M < 4){
    stop("The number of binned data should be greater than 3.")
  }
  
  xin <- dat$xin
  qin <- dat$qin

  d1 <- nrow(xin)
  d2 <- ncol(xin)
  m <- ncol(qin)
  
  if(d1 < M){
    stop("The number of binned data cannot exceed the number of observations.")
  }

  projec <- xin %*% direc
  range_of_projec <- seq(min(projec), max(projec), length.out = M)
  
  binned_xmean <- matrix(NA, M, d2)
  binned_xmean[1, ] <- xin[which.min(projec), ]
  
  binned_qmean <- matrix(NA, M, m)
  binned_qmean[1, ] <- qin[which.min(projec), ]
  
  for (l in 2:(M - 1)) {
    idx = (d1*l)%/%M
    idx_set = which(projec == sort(projec)[idx])
    binned_xmean[l, ] <- xin[idx_set[1], ]
    
    binned_qmean[l, ] <- qin[idx_set[1], ]
  }
  
  binned_xmean[M, ] <- xin[which.max(projec), ]
  binned_qmean[M, ] <- qin[which.max(projec), ]
  
  return(list(projec = projec, binned_xmean = binned_xmean, binned_qmean = binned_qmean))
}

#### CV criterion to select M
bwCV_M <- function(dat, direc, M, bw, ker = ker_gauss, lower = -Inf, upper = Inf) {
  binned_dat <- binned_data(dat, direc, M)
  xin_binned <- binned_dat$binned_xmean
  qin_binned <- binned_dat$binned_qmean
  proj_binned <- xin_binned %*% direc

  d <- nrow(qin_binned)
  m <- ncol(qin_binned)
  
  cv_err <- 0
  
  for (i in 1:d) {
    xin_eff <- xin_binned[-i, ]
    qin_eff <- qin_binned[-i, ]
    dat_cv <- list(xin = xin_eff, qin = qin_eff)
    res <- LocLin(dat_cv, direc, proj_binned[i], bw, ker = ker_gauss, lower = -Inf, upper = Inf)
    cv_err <- cv_err + mean((sort(res) - sort(qin_binned[i, ]))^2)
  }
  return(cv_err / d)
}


#### Implements the selcetion of bandw for the local fr reg and, 
#### for the  optimal choice of bandw, selects the optimal bin size
tuning <- function(dat, direc, ker = ker_gauss){
  xin = dat$xin
  qin = dat$qin

  d1 <- nrow(xin)
  d2 <- ncol(xin)
  #direc = direc_curr_i
  
  projec = xin %*% direc
  #M = missing
  #bw = missing
  xinSt = unique(sort(projec))
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
  #if ismissing(M)
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
  
  #end
  return(c(bw, M))
}


#### Main implementation function that returns 
#### the estimated direction parameter (unit vector) 
estimate_ichimura <- function(dat, bw=NULL, M=NULL, ker = ker_gauss, lower = -Inf, upper = Inf) {
  xin <- dat$xin
  qin <- dat$qin
  
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
  
  hh = tuning(dat, direc_curr_i)
  if(is.null(bw)){
    bw = hh[1]
  }
  
  if(is.null(M)){
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
                  bw, ker = ker_gauss, lower = -Inf, upper = Inf)
    err <- err + mean((sort(res) - sort(binned_dat$binned_qmean[l, ]))^2)
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
      res <- LocLin(dat, direc_test, proj_binned[l], bw, ker = ker_gauss, lower = -Inf, upper = Inf)
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

# Function to estimate parameters with multiple replications
estimate_2pred <- function(n, reps = 100, bw, M, link) {
  b <- c(4, 1.3, -2.5, 1.7)
  b0 <- normalize(b)
  d <- length(b0)
  
  est <- mclapply(
    1:reps, 
    FUN = function(x){
      dat <- generate_data2(n, b0, link)
      estimate_ichimura(dat)
      }, mc.cores = 2)
  est_signed <- lapply(est, function(x) x * sign(sum(b0 * x)))
  return(list(
    bias_calc2(est, b0, reps),
    bias_calc2(est_signed, b0, reps)
  ))
}

link <- list(function(x) x, function(x) x^2, function(x) exp(x))

for (ll in 1:3) {
  ddd <- estimate_2pred(50, reps = 20, link = link[[ll]])
  print(ddd)
}
