#' @title Local Fréchet Regression for Compositional Data
#' @description  Local Fréchet regression for compositional data with respect to the geodesic distance.
#' @param xin A vector of length n with input measurement points.
#' @param yin An n by m matrix holding the spherical data, of which the sum of squares of elements within each row is 1.
#' @param xout A vector of length k with output measurement points; Default: \code{xout = xin}.
#' @param optns A list of options control parameters specified by \code{list(name=value)}. See `Details'.
#' @details Available control options are
#' \describe{
#' \item{bw}{A scalar used as the bandwidth or \code{"CV"} (default).}
#' \item{kernel}{A character holding the type of kernel functions for local Fréchet regression for densities; \code{"rect"}, \code{"gauss"}, \code{"epan"}, \code{"gausvar"}, \code{"quar"} - default: \code{"gauss"}.}
#' }
#' @return A list containing the following components:
#' \item{xout}{Input \code{xout}.}
#' \item{yout}{A k by m matrix holding the fitted responses, of which each row is a spherical vector, corresponding to each element in \code{xout}.}
#' \item{xin}{Input \code{xin}.}
#' \item{yin}{Input \code{yin}.}
#' \item{optns}{A list of control options used.}
#' 
#' @examples
#' set.seed(1)
#' n <- 200
#' # simulate the data according to the simulation in Petersen & Müller (2019)
#' xin <- runif(n)
#' err_sd <- 0.2
#' xout <- seq(0,1,length.out = 51)
#' 
#' phi_true <- acos(xin)
#' theta_true <- pi * xin
#' ytrue <- cbind(
#'   sin(phi_true) * cos(theta_true),
#'   sin(phi_true) * sin(theta_true),
#'   cos(phi_true)
#' )
#' basis <- list(
#'   b1 = cbind(
#'     cos(phi_true) * cos(theta_true),
#'     cos(phi_true) * sin(theta_true),
#'     -sin(phi_true)
#'   ),
#'   b2 = cbind(
#'     sin(theta_true),
#'     -cos(theta_true),
#'     0
#'   )
#' )
#' yin_tg <- basis$b1 * rnorm(n, mean = 0, sd = err_sd) + 
#'   basis$b2 * rnorm(n, mean = 0, sd = err_sd)
#' yin <- t(sapply(seq_len(n), function(i) {
#'   tgNorm <- sqrt(sum(yin_tg[i,]^2))
#'   if (tgNorm < 1e-10) {
#'     return(ytrue[i,])
#'   } else {
#'     return(sin(tgNorm) * yin_tg[i,] / tgNorm + cos(tgNorm) * ytrue[i,])
#'   }
#' }))
#' 
#' res <- LocSpheReg(xin=xin, yin=yin, xout=xout, optns = list(bw = 0.15, kernel = "epan"))
#' @export 

lsr <- function(xin=NULL, yin=NULL, xout=NULL, optns=list()){
  if (is.null(xin))
    stop ("xin has no default and must be input by users.")
  if (is.null(yin))
    stop ("yin has no default and must be input by users.")
  if (is.null(xout))
    xout <- xin
  if (!is.numeric(xin))
    stop("xin should be a numerical vector or matrix.")
  if (!is.matrix(yin) | !is.numeric(yin))
    stop("yin should be a numerical matrix.")
  if (!is.numeric(xout))
    stop("xout should be a numerical vector or matrix.")
  if(is.vector(xin)){
    xin <- as.matrix(xin)
  }
  if(is.vector(xout)){
    xout <- as.matrix(xout)
  }
  if (nrow(xin)!=nrow(yin))
    stop("The number of rows in xin should be the same as the number of rows in yin.")
  if (sum(abs(rowSums(yin^2) - rep(1,nrow(yin))) > 1e-6)){
    yin = yin / sqrt(rowSums(yin^2))
    warning("Each row of yin has been standardized to enforce sum of squares equal to 1.")
  }
  
  if (is.null(optns$bw)){
    optns$bw <- "CV" #max(sort(xin)[-1] - sort(xin)[-length(xin)]) * 1.2
  }
  if (is.character(optns$bw)) {
    if (optns$bw != "CV") {
      warning("Incorrect input for optns$bw.")
    }
  } else if (!is.numeric(optns$bw)) {
    stop("Mis-specified optns$bw.")
  }
  if (length(optns$bw) > 1)
    stop("bw should be of length 1.")
  
  if (is.null(optns$kernel))
    optns$kernel <- "gauss"
  
  if (is.numeric(optns$bw)) {
    bwRange <- frechet:::SetBwRange(xin = xin, xout = xout, kernel_type = optns$kernel)
    if (optns$bw < bwRange$min | optns$bw > bwRange$max) {
      optns$bw <- "CV"
      warning("optns$bw is too small or too large; reset to be chosen by CV.")
    }
  } 
  if (optns$bw == "CV") {
    optns$bw <- bwCV_sphe(xin = xin, yin = yin, xout = xout, optns = optns)
  }
  yout <- LocSpheGeoReg(xin = xin, yin = yin, xout = xout, optns = optns)
  res <- list(xout = xout, yout = yout, xin = xin, yin = yin, optns = optns)
  class(res) <- "spheReg"
  return(res)
}

# using trust package and perturbation for initial value

LocSpheGeoReg <- function(xin, yin, xout, optns = list()) {
  k = nrow(xout)
  n = nrow(xin)
  m = ncol(yin)
  
  bw <- optns$bw
  ker <- frechet:::kerFctn(optns$kernel)
  
  yout = sapply(1:k, function(j){
    mu0 = mean(ker((xout[j] - xin) / bw))
    mu1 = mean(ker((xout[j] - xin) / bw) * (xin - xout[j]))
    mu2 = mean(ker((xout[j] - xin) / bw) * (xin - xout[j])^2)
    s = ker((xout[j] - xin) / bw) * (mu2 - mu1 * (xin - xout[j])) /
      (mu0 * mu2 - mu1^2)
    s = as.numeric(s)
    # initial guess
    y0 = colMeans(yin*s)
    y0 = y0 / l2norm(y0)
    if (sum(sapply(1:n, function(i) sum(yin[i,]*y0))[ker((xout[j] - xin) / bw)>0] > 1-1e-8)){
      #if (sum( is.infinite (sapply(1:n, function(i) (1 - sum(yin[i,]*y0)^2)^(-0.5) )[ker((xout[j] - xin) / bw)>0] ) ) +
      #   sum(sapply(1:n, function(i) 1 - sum(yin[i,] * y0)^2 < 0)) > 0){
      # return(y0)
      y0 = y0 + rnorm(3) * 1e-3
      y0 = y0 / l2norm(y0)
    }
    
    objFctn = function(y){
      # y <- y / l2norm(y)
      if ( ! isTRUE( all.equal(l2norm(y),1) ) ) {
        return(list(value = Inf))
      }
      f = mean(s * sapply(1:n, function(i) SpheGeoDist(yin[i,], y)^2))
      g = 2 * colMeans(t(sapply(1:n, function(i) SpheGeoDist(yin[i,], y) * SpheGeoGrad(yin[i,], y))) * s)
      res = sapply(1:n, function(i){
        grad_i = SpheGeoGrad(yin[i,], y)
        return((grad_i %*% t(grad_i) + SpheGeoDist(yin[i,], y) * SpheGeoHess(yin[i,], y)) * s[i])
      }, simplify = "array")
      h = 2 * apply(res, 1:2, mean)
      return(list(value=f, gradient=g, hessian=h))
    }
    res = trust::trust(objFctn, y0, 0.1, 1e5)
    # res = trust::trust(objFctn, y0, 0.1, 1)
    youtj <- res$argument / l2norm(res$argument)
    # project
    if(m == 3){
      thetaj <- acos(youtj[3])#[0, pi]
      phij <- atan2(youtj[2], youtj[1])# [-pi, pi]
      if (thetaj > pi / 2) {
        thetaj <- pi / 2
      }
      if (phij < 0) {
        phij <- 0
      } else if (thetaj > pi / 2) {
        phij <- pi / 2
      }
      youtj <- c(sin(thetaj) * cos(phij),
                 sin(thetaj) * sin(phij),
                 cos(thetaj))
    }else if(m == 4){
      # project
      theta1 = acos(youtj[4])
      theta2 = acos(youtj[3]/sqrt(1-youtj[4]^2))
      phi <- atan2(youtj[2], youtj[1])
      if (theta1 > pi / 2) {
        theta1 <- pi / 2
      }
      if(theta2 > pi/2){
        theta2 = pi/2
      }
      if (phi < 0) {
        phi <- 0
      } else if (theta2 > pi / 2) {
        phi <- pi / 2
      }
      youtj <- c(sin(theta1) * cos(phi) * sin(theta2),
                 sin(theta1) * sin(phi) * sin(theta2),
                 cos(theta1) * sin(theta2),
                 cos(theta2))
    }
    return(youtj)
  })
  return(t(yout))

}

# using CV to choose bw for local Fréchet regression on a unit hypersphere.
bwCV_sphe <- function(xin, yin, xout, optns) {
  yin <- yin[order(xin),]
  xin <- sort(xin)
  compareRange <- (xin > min(xin) + diff(range(xin))/5) & (xin < max(xin) - diff(range(xin))/5)
  
  # k-fold
  objFctn <- function(bw) {
    optns1 <- optns
    optns1$bw <- bw
    folds <- numeric(length(xin))
    n <- sum(compareRange)
    numFolds <- ifelse(n > 30, 10, sum(compareRange))
    
    tmp <- c(sapply(1:ceiling(n/numFolds), function(i)
      sample(x = seq_len(numFolds), size = numFolds, replace = FALSE)))
    tmp <- tmp[1:n]
    repIdx <- which(diff(tmp) == 0)
    for (i in which(diff(tmp) == 0)) {
      s <- tmp[i]
      tmp[i] <- tmp[i-1]
      tmp[i-1] <- s
    }
    #tmp <- cut(1:n,breaks = seq(0,n,length.out = numFolds+1), labels=FALSE)
    #tmp <- tmp[sample(seq_len(n), n)]
    
    folds[compareRange] <- tmp
    
    yout <- lapply(seq_len(numFolds), function(foldidx) {
      testidx <- which(folds == foldidx)
      res <- LocSpheGeoReg(xin = xin[-testidx], yin = yin[-testidx,], xout = xin[testidx], optns = optns1)
      res # each row is a spherical vector
    })
    yout <- do.call(rbind, yout)
    yinMatch <- yin[which(compareRange)[order(tmp)],]
    mean(sapply(1:nrow(yout), function(i) SpheGeoDist(yout[i,], yinMatch[i,])^2))
  }
  bwRange <- frechet:::SetBwRange(xin = xin, xout = xout, kernel_type = optns$ker)
  #if (!is.null(optns$bwRange)) {
  #  if (min(optns$bwRange) < bwRange$min) {
  #    message("Minimum bandwidth is too small and has been reset.")
  #  } else bwRange$min <- min(optns$bwRange)
  #  if (max(optns$bwRange) >  bwRange$min) {
  #    bwRange$max <- max(optns$bwRange)
  #  } else {
  #    message("Maximum bandwidth is too small and has been reset.")
  #  }
  #}
  res <- optimize(f = objFctn, interval = c(bwRange$min, bwRange$max))
  res$minimum
}

#'@title Geodesic distance on spheres.
#'@param y1,y2 Two unit vectors, i.e., with \eqn{L^2} norm equal to 1, of the same length.
#'@return A scalar holding the geodesic distance between \code{y1} and \code{y2}.
#'@examples
#'d <- 3
#'y1 <- rnorm(d)
#'y1 <- y1 / sqrt(sum(y1^2))
#'y2 <- rnorm(d)
#'y2 <- y2 / sqrt(sum(y2^2))
#'dist <- SpheGeoDist(y1,y2)
#'@export

SpheGeoDist <- function(y1,y2) {
  if (abs(length(y1) - length(y2)) > 0) {
    stop("y1 and y2 should be of the same length.")
  }
  if ( !isTRUE( all.equal(l2norm(y1),1) ) ) {
    stop("y1 is not a unit vector.")
  }
  if ( !isTRUE( all.equal(l2norm(y2),1) ) ) {
    stop("y2 is not a unit vector.")
  }
  y1 = y1 / l2norm(y1)
  y2 = y2 / l2norm(y2)
  if (sum(y1 * y2) > 1){
    return(0)
  } else if (sum(y1*y2) < -1){
    return(pi)
  } else return(acos(sum(y1 * y2)))
}

#' Compute gradient w.r.t. y of the geodesic distance \eqn{\arccos(x^\top y)} on a unit hypersphere
#' @param x,y Two unit vectors.
#' @return A vector holding radient w.r.t. \code{y} of the geodesic distance between \code{x} and \code{y}.
#' @export
SpheGeoGrad <- function(x,y) { 
  tmp <- 1 - sum(x * y) ^ 2
  return(- (tmp) ^ (-0.5) * x)
  # if (tmp < tol) {
  #   return(- Inf * x)
  # } else {
  #   return(- (tmp) ^ (-0.5) * x)
  # }
}

#' Hessian \eqn{\partial^2/\partial y \partial y^\top} of the geodesic distance \eqn{\arccos(x^\top y)} on a unit hypersphere
#' @param x,y Two unit vectors.
#' @return A Hessian matrix.
#' @export
SpheGeoHess <- function(x,y) { #,tol = 1e-10){
  return(- sum(x * y) * (1 - sum(x * y) ^ 2) ^ (-1.5) * x %*% t(x))
}

# L2 norm
l2norm <- function(x){
  #sqrt(sum(x^2))
  as.numeric(sqrt(crossprod(x)))
}