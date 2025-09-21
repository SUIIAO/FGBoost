#' @title Global Fréchet Regression for Compositional Data
#' @description  Global Fréchet regression for compositional data with respect to the geodesic distance.
#' @param xin A vector of length \eqn{n} or an \eqn{n}-by-\eqn{p} matrix with input measurement points.
#' @param yin An \eqn{n}-by-\eqn{m} matrix holding the spherical data, of which the sum of squares of elements within each row is 1.
#' @param xout A vector of length \eqn{k} or an \eqn{k}-by-\eqn{p}  with output measurement points; Default: the same grid as given in \code{xin}.
#' @return A list containing the following components:
#' \item{xout}{Input \code{xout}.}
#' \item{yout}{A \eqn{k}-by-\eqn{m} matrix holding the fitted responses, of which each row is a spherical vector, corresponding to each element in \code{xout}.}
#' \item{xin}{Input \code{xin}.}
#' \item{yin}{Input \code{yin}.}
#' 
#' @examples
#' n <- 101
#' xin <- seq(-1,1,length.out = n)
#' theta_true <- rep(pi/2,n)
#' phi_true <- (xin + 1) * pi / 4
#' ytrue <- apply( cbind( 1, phi_true, theta_true ), 1, pol2car )
#' yin <- t( ytrue )
#' xout <- xin
#' res <- GloSpheReg(xin=xin, yin=yin, xout=xout)
#' @references
#' \cite{Petersen, A., & Müller, H.-G. (2019). "Fréchet regression for random objects with Euclidean predictors." The Annals of Statistics, 47(2), 691--719.}
#' @export 

gsr <- function(xin=NULL, yin=NULL, xout=NULL){
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
  
  yout <- GloSpheGeoReg(xin = xin, yin = yin, xout = xout)
  res <- list(xout = xout, yout = yout, xin = xin, yin = yin)
  class(res) <- "spheReg"
  return(res)
}

# using trust package and perturbation for initial value

GloSpheGeoReg <- function(xin, yin, xout) {
  k = nrow(xout)
  n = nrow(xin)
  m = ncol(yin)
  
  xbar <- colMeans(xin)
  Sigma <- cov(xin) * (n-1) / n
  invSigma <- solve(Sigma)
  
  yout = sapply(1:k, function(j){
    s <- 1 + t(t(xin) - xbar) %*% invSigma %*% (xout[j,] - xbar)
    s <- as.vector(s)
    
    # initial guess
    y0 = colMeans(yin*s)
    y0 = y0 / l2norm(y0)
    if (any(sapply(1:n, function(i)
      isTRUE(all.equal(sum(
        yin[i, ] * y0
      ), 1))))) {
      # if (sum(sapply(1:n, function(i) sum(yin[i,]*y0)) > 1-1e-8)){
      #if (sum( is.infinite (sapply(1:n, function(i) (1 - sum(yin[i,]*y0)^2)^(-0.5) )[ker((xout[j] - xin) / bw)>0] ) ) +
      #   sum(sapply(1:n, function(i) 1 - sum(yin[i,] * y0)^2 < 0)) > 0){
      y0[1] = y0[1] + 1e-3
      y0 = y0 / l2norm(y0)
    }
    
    objFctn = function(y){
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
    youtj <- res$argument / l2norm(res$argument)
    if(m == 3){
      # project
      thetaj <- acos(youtj[3]) # [0, pi]
      phij <- atan2(youtj[2], youtj[1]) # [-pi, pi]
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