# Recursive function to build a decision tree with specified depth
build_tree = function(X, residuals, y, F_cur, depth, learning_rate, max_depth, optns = list()) {
  # If we've reached the maximum depth, return the mean of the residuals
  if (depth >= max_depth) {
    if (optns$type == 'compositional') {
      n = length(residuals)
      
      return(list(predict = list(
        start = brct(y = lapply(residuals, function(x) x$start), optns = optns),
        end = brct(y = lapply(residuals, function(x) x$end), optns = optns)),
        num = n))
    }else if(optns$type %in% c("measure", "laplacian")){
      n = length(residuals)
      
      return(list(predict = list(
        start = Reduce("+", lapply(residuals, function(x) x$start))/n,
        end = Reduce("+", lapply(residuals, function(x) x$end))/n),
        num = n))
      
    }else{
      return(list(predict = mean(residuals),
                  num = length(residuals)))
    }
  }
  
  # Otherwise, find the best split
  n = nrow(X)
  
  if(optns$type %in% c("compositional", "measure", "laplacian") & optns$impurity == "MVAR"){
    start = lapply(residuals, function(x) x$start)
    end = lapply(residuals, function(x) x$end)
    
    pdx <- pdy <- pdxy <- matrix(0, nrow = n, ncol = n)
    for(i in 1:n) {
      for(j in 1:n) {
        pdxy[i, j] <- distFun(start[[i]], end[[j]], optns)
      }
    }
    for(i in 1:(n - 1)) {
      for(j in (i + 1):n) {
        pdx[i, j] <- pdx[j, i] <- distFun(start[[i]], start[[j]], optns)
        pdy[i, j] <- pdy[j, i] <- distFun(end[[i]], end[[j]], optns)
      }
    }
  }
  
  paras = NULL
  for (j in 1:ncol(X)) {
    paras = rbind(paras, data.frame(j = j, thresholds = quantile(X[, j], probs = seq(1/34, 33/34, length.out = 33))))
  }
  if(optns$ncores == 1){
    result = foreach::foreach(para = t(paras), .export = c("brct","pt","mse_loss","plcm","lcm","gcd")) %do% {
      j = para[1]
      threshold = para[2]
      
      left_indices = X[, j] <= threshold
      right_indices = X[, j] > threshold
      if (sum(left_indices) < optns$min_samples_per_leaf | sum(right_indices) < optns$min_samples_per_leaf) {
        return(list(loss = data.frame(j = j, threshold = threshold, loss = Inf)))
      }
      
      y_left = y[left_indices]
      y_right = y[right_indices]
      
      left_start = lapply(residuals[left_indices], function(x) x$start)
      left_end = lapply(residuals[left_indices], function(x) x$end)
      right_start = lapply(residuals[right_indices], function(x) x$start)
      right_end = lapply(residuals[right_indices], function(x) x$end)
      
      left_idx = which(left_indices)
      right_idx = which(right_indices)
      
      if(optns$impurity == "MSE"){
        left_mean = list(
          start = brct(y = left_start, optns = optns),
          end = brct(y = left_end, optns = optns)
        )
        
        right_mean = list(
          start = brct(y = right_start, optns = optns),
          end = brct(y = right_end, optns = optns)
        )
        F_cur_left = lapply(1:length(left_idx), function(i){
          return(pt(alpha = left_mean$start, 
                    beta = left_mean$end,
                    omega = F_cur[[left_idx[i]]], 
                    learning_rate = learning_rate,
                    optns = optns))
        })
        
        F_cur_right = lapply(1:length(right_idx), function(i){
          return(pt(alpha = right_mean$start, 
                    beta = right_mean$end,
                    omega = F_cur[[right_idx[i]]], 
                    learning_rate = learning_rate,
                    optns = optns))
        })
        
        loss = sum(c(sapply(1:length(y_left), function(i){
          mse_loss(y_left[[i]],F_cur_left[[i]],optns)
        }),
        sapply(1:length(y_right), function(i){
          mse_loss(y_right[[i]],F_cur_right[[i]],optns)
        }))) - sum(sapply(1:length(y), function(i){
          mse_loss(y[[i]],F_cur[[i]],optns)
        }))
      }
      
      return(list(loss = data.frame(j = j, threshold = threshold, loss = loss), 
                  F_cur_left = F_cur_left, F_cur_right = F_cur_right))
    }
  }else{
    result = foreach::foreach(para = t(paras), .export = c("brct","pt","mse_loss","plcm","lcm","gcd")) %dopar% {
      j = para[1]
      threshold = para[2]
      
      left_indices = X[, j] <= threshold
      right_indices = X[, j] > threshold
      if (sum(left_indices) < optns$min_samples_per_leaf | sum(right_indices) < optns$min_samples_per_leaf) {
        return(list(loss = data.frame(j = j, threshold = threshold, loss = Inf)))
      }
      
      y_left = y[left_indices]
      y_right = y[right_indices]
      
      left_start = lapply(residuals[left_indices], function(x) x$start)
      left_end = lapply(residuals[left_indices], function(x) x$end)
      right_start = lapply(residuals[right_indices], function(x) x$start)
      right_end = lapply(residuals[right_indices], function(x) x$end)
      
      left_idx = which(left_indices)
      right_idx = which(right_indices)
      
      if(optns$impurity == "MSE"){
        left_mean = list(
          start = brct(y = left_start, optns = optns),
          end = brct(y = left_end, optns = optns)
        )
        
        right_mean = list(
          start = brct(y = right_start, optns = optns),
          end = brct(y = right_end, optns = optns)
        )
        F_cur_left = lapply(1:length(left_idx), function(i){
          return(pt(alpha = left_mean$start, 
                    beta = left_mean$end,
                    omega = F_cur[[left_idx[i]]], 
                    learning_rate = learning_rate,
                    optns = optns))
        })
        
        F_cur_right = lapply(1:length(right_idx), function(i){
          return(pt(alpha = right_mean$start, 
                    beta = right_mean$end,
                    omega = F_cur[[right_idx[i]]], 
                    learning_rate = learning_rate,
                    optns = optns))
        })
        
        loss = sum(c(sapply(1:length(y_left), function(i){
          mse_loss(y_left[[i]],F_cur_left[[i]],optns)
        }),
        sapply(1:length(y_right), function(i){
          mse_loss(y_right[[i]],F_cur_right[[i]],optns)
        }))) - sum(sapply(1:length(y), function(i){
          mse_loss(y[[i]],F_cur[[i]],optns)
        }))
      }
      
      return(list(loss = data.frame(j = j, threshold = threshold, loss = loss), 
                  F_cur_left = F_cur_left, F_cur_right = F_cur_right))
    }
  }
  
  
  loss = lapply(result, function(x) x$loss)
  loss = do.call(rbind, loss)
  
  if(all(loss$loss == Inf)){
    n = length(residuals)
    return(list(predict = list(
      start = brct(y = lapply(residuals, function(x) x$start), optns = optns),
      end = brct(y = lapply(residuals, function(x) x$end), optns = optns)),
      num = n
    ))
  }else{
    idx_best = which.min(loss$loss)
    best_loss = loss$loss[idx_best]
    best_split = list(j = loss$j[idx_best], threshold = loss$threshold[idx_best])
    best_F_cur_left = result[[idx_best]]$F_cur_left
    best_F_cur_right = result[[idx_best]]$F_cur_right
  }
  
  
  # Split the data and build the left and right subtrees recursively
  left_indices = X[, best_split$j] <= best_split$threshold
  right_indices = X[, best_split$j] > best_split$threshold
  
  left_idx = which(left_indices)
  right_idx = which(right_indices)
  
  left_tree = build_tree(X = X[left_indices, , drop = FALSE], residuals = residuals[left_indices],
                          y = y[left_indices], F_cur = best_F_cur_left,
                          depth = depth + 1, learning_rate, max_depth, optns)
  right_tree = build_tree(X[right_indices, , drop = FALSE], residuals[right_indices], 
                           y = y[right_indices], F_cur = best_F_cur_right, 
                           depth + 1, learning_rate, max_depth, optns)
  
  return(list(j = best_split$j, threshold = best_split$threshold,
              left = left_tree, right = right_tree,
              num = n))
}