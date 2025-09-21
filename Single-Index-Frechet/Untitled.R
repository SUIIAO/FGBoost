# Bug
source("Single-Index-Frechet/SIdxCovReg.R")
set.seed(1)

n = 100

# Set Parameters
X = matrix(c(runif(n, -1, 1), 
             runif(n, -1, 1),
             runif(n, 1, 2),
             
             rgamma(n, 3, 1),
             rgamma(n, 4, 1),
             rgamma(n, 5, 1),
             
             rbinom(n, 1, 0.2),
             rbinom(n, 1, 0.3),
             rbinom(n, 1, 0.5)
), 
n) 

y = lapply(1:n, function(i){
  a = 2 * sin(pi*X[i,1])^2*X[i,7] + cos(pi*X[i,2])^2*(1-X[i,7])
  b = X[i,4]*X[i,8]+X[i,5]*(1-X[i,8])
  
  Vec = -rbeta(m*(m-1)/2, shape1 = a, shape2 = b)
  temp = matrix(0, nrow = m, ncol = m)
  temp[lower.tri(temp)] = Vec
  temp <- temp + t(temp)
  diag(temp) = -colSums(temp)
  return(temp)
})

y_mat = array(0, c(m, m, n))
for(j in 1:n){
  y_mat[,,j] = y[[j]]
}

res_sid = SIdxCovReg(xin = as.matrix(X), Min = y_mat)
