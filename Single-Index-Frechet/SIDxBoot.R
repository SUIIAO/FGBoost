DenBoot_est <- function(est, xin, qin, reps, bw, M){
  n = nrow(xin)
  d = length(fit)
  samp_ind = sample(1:n, n, replace = T)
  ## Consider using different resampling strategy
  ## i.e., should resample reps time
  xin_resamp = xin[samp_ind, ]
  qin_resamp = qin[samp_ind, ]
  
  b_est <- lapply(
    1:reps, 
    FUN = function(x){
      SIdxDenReg(xin_resamp, qin_resamp, bw, M)
    })
  
  est_signed <- lapply(b_est, function(x) {x[2:d] * sign(sum(est[2:d] * x[2:d]))})
  return(est_signed)
}

## Test
tt <- DenBoot_est(est = fit, xin = x_in, qin = y_in, reps = 5, bw = 0.2, M = 5)
cova_boot  = var(do.call(rbind, tt))
M = 5
test_stat_boot = sapply(tt, function(x){
    M * t(x - fit[2:d]) %*% solve(cova_boot, x - fit[2:d])
    })
test_stat = c(M * t(fit[2:d] - b0[2:d]) %*% solve(cova_boot, fit[2:d] - b0[2:d]))
p_val = mean(test_stat_boot >= test_stat)
p_val
qchisq(0.95, df = 3) # critical value?
