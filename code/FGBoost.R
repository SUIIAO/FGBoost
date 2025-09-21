#' @title Frechet Geodesic Boosting (FGBoost)
#' @description Implements the Frechet Geodesic Boosting algorithm for metric-space-valued responses with Euclidean predictors.
#' @param X An n by p matrix or data frame of predictors.
#' @param y A list of n observations, where each element represents the metric space-valued response.
#' @param n_estimators An integer, the number of boosting stages (trees) to perform.
#' @param learning_rate A float, the learning rate that shrinks the contribution of each tree.
#' @param max_depth An integer, the maximum depth of the individual regression trees.
#' @param optns A list of optional parameters specified as \code{list(name = value)}. See `Details` for available control options.
#' @details The control options available in \code{optns} are:
#' \describe{
#'   \item{type}{A character string specifying the type of data. Supported types are 'measure' (for probability measures), 'compositional', or 'laplacian'. This option is crucial as it dictates how distances, barycenters, and parallel transports are computed. Required.}
#'   \item{seed}{An integer, the random seed for reproducibility. If \code{NULL} (default), a random seed is not explicitly set by this function, though internal processes might use their own defaults.}
#'   \item{ncores}{An integer, the number of cores to use for parallel processing. Default is 1. (Note: Current implementation in the provided code has parallel processing parts commented out).}
#'   \item{validation_fraction}{A numeric value between 0 and 1. The proportion of the training data to set aside as a validation set for monitoring performance and for early stopping. If \code{NULL} (default), no validation set is used and early stopping is disabled.}
#'   \item{early_stopping_rounds}{An integer. If \code{validation_fraction} is set, training will stop if the validation error does not improve for this many consecutive rounds. If \code{NULL} (default) and \code{validation_fraction} is set, early stopping is not performed based on a fixed number of rounds but monitoring will still occur.}
#' }
#' @return A list containing:
#' \item{F_0}{The initial model, typically the barycenter of the training responses.}
#' \item{final_predictions}{A list of the final fitted values for the training data \code{X}.}
#' \item{trees}{A list containing all the fitted decision trees from each boosting iteration.}
#' \item{err_train}{A numeric vector of training errors (MSE based on the specified \code{optns$type}) at each boosting iteration.}
#' \item{err_valid}{A numeric vector of validation errors at each boosting iteration. Only returned if \code{optns$validation_fraction} is not \code{NULL}.}
#' \item{learning_rate}{The learning rate used in the boosting process.}
#' \item{optns}{The list of options that were used to run the function.}
#' @export

# Gradient Boosting Algorithm with Trees of Depth 3
FGBoost = function(X, y, n_estimators, learning_rate, max_depth, optns = list()) {
  if(!is.null(optns$seed)){
    set.seed(optns$seed) # Set random seed for reproducibility
  }
  
  if(is.null(optns$ncores)){
    optns$ncores = 1 # Default to 1 core if not specified
    ncores = 1
  }else{
    ncores = optns$ncores # Use specified number of cores
  }
  
  if(is.null(optns$early_stopping_rounds)){
    early_stopping_rounds = NULL # No early stopping if not specified
  }else{
    early_stopping_rounds = optns$early_stopping_rounds # Use specified early stopping rounds
  }
  
  err_train = NULL # Initialize training error log
  
  # Split data into training and validation sets if validation_fraction is provided
  if(!is.null(optns$validation_fraction)){
    n = nrow(X) # Total number of observations
    all_idx = 1:n # All indices
    # Sample indices for the training set
    train_idx = sample(all_idx, round((1-optns$validation_fraction) * n))
    # Indices for the validation set are those not in the training set
    valid_idx = setdiff(all_idx, train_idx)
    
    n_train = length(train_idx) # Number of training observations
    X_train = X[train_idx, ]    # Training predictors
    y_train = y[train_idx]      # Training responses
    
    n_valid = length(valid_idx) # Number of validation observations
    X_valid = X[valid_idx, ]    # Validation predictors
    y_valid = y[valid_idx]      # Validation responses
    
    err_valid = NULL # Initialize validation error log
  }else{
    # If no validation fraction, use all data for training
    n_train = nrow(X)
    X_train = X
    y_train = y
  }

  # Initialize model: F_0 is the barycenter of the training responses
  F_0 = brct(y = y_train, optns = optns)
  # Initialize current predictions F for each training observation with F_0
  F = lapply(1:n_train, function(i){
    return(F_0)
  })
  # Initialize current predictions F_valid for each validation observation if applicable
  if(!is.null(optns$validation_fraction)){
    F_valid = lapply(1:n_valid, function(i){
      return(F_0)
    })
  }
  
  if(ncores > 1){
    cl = parallel::makeCluster(ncores) # parallel processing setup
    doSNOW::registerDoSNOW(cl)        # register parallel backend
  }
  
  trees = list() # List to store all fitted trees
  non_improve_round = 0 # Counter for early stopping: rounds without improvement
  min_valid = Inf       # Initialize minimum validation error for early stopping
  cat("Building the Frechet Geodesic Boosting trees...\n")
  start_time = Sys.time()  # Record the start time for training duration
  
  # Gradient Boosting Loop: iterate n_estimators times
  for (m in 1:n_estimators) {
    # Compute residuals: difference between true responses and current predictions
    residuals = compute_residuals(y_train, F, optns = optns)
    
    # Fit a decision tree to the residuals
    tree = build_tree(X = X_train, residuals = residuals, y = y_train, F_cur = F, depth = 0, 
                      learning_rate = learning_rate, max_depth = max_depth, optns = optns)
    
    # Get predictions from the newly fitted tree for training data
    tree_pred = predict_tree(tree, X_train, optns)
    # Get predictions for validation data if applicable
    if(!is.null(optns$validation_fraction)){
      tree_pred_valid = predict_tree(tree, X_valid, optns)
    }
    
    # Update the ensemble predictions for the training set
    # Each F[[i]] is updated by moving along the geodesic from F[[i]] towards the tree's prediction
    F = lapply(1:length(F), function(i){
      return(pt(alpha = tree_pred[[i]]$start, # Current prediction F[[i]]
                beta = tree_pred[[i]]$end,   # Target (based on tree leaf value)
                omega = F[[i]],              # Current prediction F[[i]] (transported point is around this)
                learning_rate = learning_rate,
                optns = optns))
    })
    
    # Calculate and store the mean training error for the current iteration
    err_train = c(err_train, mean(sapply(1:length(F), function(i) {
      mse_loss(y_train[[i]], F[[i]], optns)
    })))
    
    # If using a validation set, update predictions and check for early stopping
    if(!is.null(optns$validation_fraction)){
      # Update ensemble predictions for the validation set
      tree_pred_valid = predict_tree(tree, X_valid, optns) # Redundant if already predicted above, can be optimized
      F_valid = lapply(1:length(F_valid), function(i){
        return(pt(alpha = tree_pred_valid[[i]]$start,
                  beta = tree_pred_valid[[i]]$end,
                  omega = F_valid[[i]], 
                  learning_rate = learning_rate,
                  optns = optns))
      })
      # Calculate mean validation error for the current iteration
      err_valid_m = mean(sapply(1:n_valid, function(i) mse_loss(y_valid[[i]], F_valid[[i]], optns)))
      err_valid = c(err_valid, err_valid_m)
      
      # Early stopping logic
      if(!is.null(early_stopping_rounds)){
        if(min_valid > err_valid_m){ # If current validation error is lower
          min_valid = err_valid_m   # Update minimum validation error
          non_improve_round = 0     # Reset counter for rounds without improvement
        }else{
          non_improve_round = non_improve_round + 1 # Increment counter
        }
        # If no improvement for `early_stopping_rounds`
        if(non_improve_round >= early_stopping_rounds){
          cat("\nEarly stopping at iteration ", m, " with validation loss ", min_valid, "\n")
          break # Exit the boosting loop
        }
      }
    }

    # Store the fitted tree
    trees[[m]] = tree
    
    # Calculate elapsed time
    elapsed_time = as.numeric(Sys.time() - start_time, units = "secs")
    
    # Estimate remaining time
    remaining_steps = n_estimators - m
    est_total_time = elapsed_time / m * n_estimators
    time_left = est_total_time - elapsed_time
    
    # Format remaining time for display
    mins_left = floor(time_left / 60)
    secs_left = round(time_left %% 60)
    
    # Update and print progress bar
    progress = sprintf("|%-50s| %d %% ~ %2d m %2d s", 
                        paste(rep("+", floor(m/n_estimators*50)), collapse = ""), 
                        round(m / n_estimators * 100), 
                        mins_left, secs_left)
    
    cat("\r", progress)
    flush.console() # Ensure progress bar updates immediately
    # setTxtProgressBar(pb, m) # Alternative progress bar (commented out)
  }
  # Calculate total training time
  temps = as.numeric(Sys.time() - start_time, units = "secs")
  
  # Format total time for display
  mins_total = floor(temps / 60)
  secs_total = round(temps %% 60)
  # Final progress bar display showing 100% completion and total time
  progress = sprintf("|%-50s| %d %% ~ %2d m %2d s", 
                      paste(rep("+", floor(m/m*50)), collapse = ""), # m/m ensures 100% if loop finished, or current m if early stopped
                      round(m / m * 100), 
                     mins_total, secs_total)
  
  cat("\r", progress)
  flush.console()

  # parallel::stopCluster(cl) # Commented out: stop parallel cluster
  
  # Return model results
  if(is.null(optns$validation_fraction)){
    # Return list without validation error if no validation set was used
    return(list(F_0 = F_0, final_predictions = F, trees = trees, err_train = err_train, 
                learning_rate = learning_rate, optns = optns))
  }else{
    # Return list including validation error
    return(list(F_0 = F_0, final_predictions = F, trees = trees, err_train = err_train, err_valid = err_valid, 
                learning_rate = learning_rate, optns = optns))
  }
}

# barycenter of a list of random objects
brct = function(y, optns, w = rep(1 / length(y), length(y))) {# y is a list
  n <- length(y)
  if (optns$type == 'measure') {
    N <- sapply(y, length)
    y <- lapply(1:n, function(i) {
      sort(y[[i]])
    }) # sort observed values
    M <- min(plcm(N), n * max(N), 5000) # least common multiple of N_i
    yM <- t(sapply(1:n, function(i) {
      residual <- M %% N[i]
      if (residual) {
        sort(c(rep(y[[i]], each = M %/% N[i]), y[[i]][1:residual]))
      } else {
        rep(y[[i]], each = M %/% N[i])
      }
    })) # n by M
    brct <- apply(yM, 2, weighted.mean, w)
  } else if (optns$type == 'compositional') {
    mfd <- structure(1, class = 'Sphere')
    yM <- matrix(unlist(y), ncol = n) # 3 by n
    brct <- c(manifold::frechetMean(mfd = mfd, X = yM, weight = w, maxit = 1e04)) # RFPCA:::frechetMean.Sphere
  } else if (optns$type %in% c('laplacian', 'correlation', 'covariance')) {
    brct <- purrr::reduce(lapply(1:n, function(i) w[i] * y[[i]]), `+`)
  }
  brct
}

# parallel transport
pt <- function(alpha, beta, omega, learning_rate, optns) {
  if (optns$type == 'measure') {
    zeta <- pracma::spinterp(x = alpha, y = (beta-alpha)*learning_rate + alpha, xp = omega)
  } else if (optns$type == 'compositional') {
    v <- beta - sum(alpha * beta) * alpha
    pwv <- v - sum(omega * v) * omega
    pwv <- pwv / sqrt(sum(pwv^2))
    at <- learning_rate * acos(sum(alpha * beta))
    zeta <- cos(at) * omega + sin(at) * pwv
  } else if (optns$type == 'laplacian'){
    zeta <- omega + (beta - alpha) * learning_rate
  }
  zeta
}

# Mean Squared Error Loss
mse_loss <- function(y_true, y_pred, optns = list()) {
  if (optns$type == "compositional") {
    prod = sum(y_true * y_pred)
    if(prod > 1){
      return(acos(1)^2)
    }else{
      return(acos(prod)^2)
    }
  }else if(optns$type == "measure"){
    return(mean((y_true - y_pred)^2))    # W2 distance
  }else if(optns$type == "laplacian"){
    return(sum((y_true - y_pred)^2))     # Frobenius norm
  }else{
    return(mean((y_true - y_pred)^2))    # Euclidean distance
  }
}

# Compute Residuals
compute_residuals <- function(y, y_pred, optns = list()) {
  if (optns$type %in% c('compositional', "measure", 'laplacian')) {
    res = lapply(1:length(y), function(i){
      l = list()
      l$start = y_pred[[i]]
      l$end = y[[i]]
      return(l)
    })
    return(res)
  }else{
    return(y - y_pred)
  }
}