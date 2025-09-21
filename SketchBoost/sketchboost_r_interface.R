# SketchBoost R Interface
# This script provides R functions to use SketchBoost via Python

# Load required libraries
library(reticulate)

# Set up Python environment (adjust path as needed)
# You may need to specify the path to your Python environment
# use_python("/path/to/your/python", required = TRUE)
# Or use the virtual environment:
# use_virtualenv("/Users/suiiao/Google Drive/XGBoost/NeurIPS_submission_rebuttal/sketchboost-paper-main/sb_env")

# Import the SketchBoost module
# Make sure the working directory contains sketchboost_module.py
setwd("/Users/suiiao/Google Drive/XGBoost/NeurIPS_submission_rebuttal/SketchBoost")
sketchboost <- import("sketchboost_module")

#' Fit SketchBoost Model
#'
#' @param X_train Training features (matrix or data.frame)
#' @param y_train Training targets (matrix or data.frame)
#' @param sketch_dim Sketching dimension (default: 5)
#' @param sketch_method Sketching method: "proj" (projection), "rand" (random), "topk" (top-k) (default: "proj")
#' @param n_estimators Number of boosting rounds (default: 100)
#' @param max_depth Maximum tree depth (default: 4)
#' @param learning_rate Learning rate (default: 0.1)
#' @param random_state Random seed (default: 42)
#'
#' @return Fitted SketchBoost model
#' @export
fit_sketchboost <- function(X_train, y_train, sketch_dim = 5, sketch_method = "proj",
                            n_estimators = 100, max_depth = 4, learning_rate = 0.1, 
                            random_state = 42) {
  
  # Validate inputs
  if (!sketch_method %in% c("proj", "rand", "topk", "random")) {
    stop("sketch_method must be one of: 'proj', 'rand', 'topk', 'random'")
  }
  
  # Convert R objects to appropriate format
  X_train <- as.matrix(X_train)
  y_train <- as.matrix(y_train)
  
  cat(sprintf("Fitting SketchBoost: %d samples, %d features -> %d outputs\n", 
              nrow(X_train), ncol(X_train), ncol(y_train)))
  cat(sprintf("Parameters: sketch_dim=%d, method=%s, trees=%d\n", 
              sketch_dim, sketch_method, n_estimators))
  
  # Create and fit the model
  model <- sketchboost$SketchBoost(
    sketch_dim = as.integer(sketch_dim),
    sketch_method = sketch_method,
    n_estimators = as.integer(n_estimators),
    max_depth = as.integer(max_depth),
    learning_rate = learning_rate,
    random_state = as.integer(random_state)
  )
  
  # Fit the model
  model$fit(X_train, y_train)
  
  return(model)
}

#' Make Predictions with SketchBoost Model
#'
#' @param model Fitted SketchBoost model (from fit_sketchboost)
#' @param X_test Test features to predict on
#'
#' @return Matrix of predictions (y_test predictions)
#' @export
predict_sketchboost <- function(model, X_test) {
  X_test <- as.matrix(X_test)
  y_test_pred <- model$predict(X_test)
  return(y_test_pred)
}

#' Train SketchBoost and Predict in One Step
#'
#' @param X_train Training features (matrix or data.frame)
#' @param y_train Training targets (matrix or data.frame)
#' @param X_test Test features to predict on
#' @param sketch_dim Sketching dimension (default: 5)
#' @param sketch_method Sketching method: "proj", "rand", "topk" (default: "proj")
#' @param n_estimators Number of boosting rounds (default: 100)
#' @param max_depth Maximum tree depth (default: 4)
#' @param learning_rate Learning rate (default: 0.1)
#' @param random_state Random seed (default: 42)
#'
#' @return Matrix of predictions (y_test predictions)
#' @export
sketchboost_predict <- function(X_train, y_train, X_test, sketch_dim = 5, 
                                sketch_method = "proj", n_estimators = 100, 
                                max_depth = 4, learning_rate = 0.1, random_state = 42) {
  
  # Fit the model
  model <- fit_sketchboost(
    X_train = X_train,
    y_train = y_train,
    sketch_dim = sketch_dim,
    sketch_method = sketch_method,
    n_estimators = n_estimators,
    max_depth = max_depth,
    learning_rate = learning_rate,
    random_state = random_state
  )
  
  # Make predictions
  y_test_pred <- predict_sketchboost(model, X_test)
  
  return(y_test_pred)
}

#' Generate Laplacian Data for Testing
#'
#' @param n_samples Number of samples
#' @param n_features Number of features
#' @param n_outputs Number of outputs
#' @param loc Location parameter for Laplacian distribution
#' @param scale Scale parameter for Laplacian distribution
#'
#' @return List with X (features) and y (targets)
#' @export
generate_laplacian_data <- function(n_samples = 1000, n_features = 12, n_outputs = 10,
                                    loc = 0, scale = 1) {
  
  # Generate features from Laplacian distribution
  # Note: R doesn't have a built-in Laplacian, so we'll use a simple approximation
  # or you can install the 'extraDistr' package for rlaplace()
  
  # Using difference of exponentials to approximate Laplacian
  set.seed(42)
  u1 <- matrix(rexp(n_samples * n_features), n_samples, n_features)
  u2 <- matrix(rexp(n_samples * n_features), n_samples, n_features)
  X <- loc + scale * (u1 - u2)
  
  # Generate random coefficient matrix
  coef <- matrix(runif(n_features * n_outputs), n_features, n_outputs)
  
  # Generate targets with noise
  y <- X %*% coef + matrix(rnorm(n_samples * n_outputs, 0, 0.1), n_samples, n_outputs)
  
  return(list(X = X, y = y))
}

#' Demo function to show SketchBoost usage with different methods
#'
#' @param n_samples Number of samples for demo data
#' @param sketch_dims Vector of sketch dimensions to test
#' @param sketch_methods Vector of sketch methods to test
#'
#' @return Results summary
#' @export
demo_sketchboost <- function(n_samples = 1000, sketch_dims = c(3, 5, 7), 
                             sketch_methods = c("proj", "rand", "topk")) {
  
  cat("SketchBoost CPU Demo\n")
  cat("====================\n\n")
  
  # Generate demo data
  cat("Generating Laplacian data...\n")
  data <- generate_laplacian_data(n_samples = n_samples)
  
  # Split into train/test
  train_idx <- sample(nrow(data$X), 0.8 * nrow(data$X))
  X_train <- data$X[train_idx, ]
  y_train <- data$y[train_idx, ]
  X_test <- data$X[-train_idx, ]
  y_test <- data$y[-train_idx, ]
  
  cat(sprintf("Data shapes: X_train (%d, %d), y_train (%d, %d)\n", 
              nrow(X_train), ncol(X_train), nrow(y_train), ncol(y_train)))
  cat(sprintf("             X_test (%d, %d), y_test (%d, %d)\n\n", 
              nrow(X_test), ncol(X_test), nrow(y_test), ncol(y_test)))
  
  results <- list()
  
  # Test different sketch methods and dimensions
  for (method in sketch_methods) {
    cat(sprintf("\n=== Testing sketch method: %s ===\n", method))
    
    for (dim in sketch_dims) {
      cat(sprintf("  Sketch dimension: %d\n", dim))
      
      tryCatch({
        # Fit model
        model <- fit_sketchboost(
          X_train = X_train,
          y_train = y_train,
          sketch_dim = dim,
          sketch_method = method,
          n_estimators = 50,  # Reduced for demo speed
          verbose = FALSE
        )
        
        # Make predictions
        y_pred <- predict_sketchboost(model, X_test)
        
        # Calculate metrics
        mse <- mean((y_test - y_pred)^2)
        ss_res <- sum((y_test - y_pred)^2)
        ss_tot <- sum((y_test - mean(y_test))^2)
        r2 <- 1 - (ss_res / ss_tot)
        
        cat(sprintf("    MSE: %.6f, R²: %.6f\n", mse, r2))
        
        results[[paste0(method, "_dim_", dim)]] <- list(
          method = method,
          sketch_dim = dim,
          mse = mse,
          r2 = r2,
          model = model
        )
        
      }, error = function(e) {
        cat(sprintf("    Error: %s\n", e$message))
        results[[paste0(method, "_dim_", dim)]] <- list(
          method = method,
          sketch_dim = dim,
          error = e$message
        )
      })
    }
  }
  
  # Summary
  cat("\n=== Summary ===\n")
  successful_results <- results[sapply(results, function(x) !is.null(x$r2))]
  
  if (length(successful_results) > 0) {
    best_result <- successful_results[[which.max(sapply(successful_results, function(x) x$r2))]]
    cat(sprintf("Best result: %s with dim=%d (R² = %.6f)\n", 
                best_result$method, best_result$sketch_dim, best_result$r2))
  }
  
  cat("\nDemo completed!\n")
  return(results)
}

#' Compare CPU SketchBoost vs XGBoost Fallback
#'
#' @param X_train Training features
#' @param y_train Training targets
#' @param X_test Test features
#' @param y_test Test targets
#' @param sketch_dim Sketching dimension
#'
#' @return Comparison results
#' @export
compare_implementations <- function(X_train, y_train, X_test, y_test, sketch_dim = 5) {
  
  cat("Comparing CPU SketchBoost vs XGBoost Fallback\n")
  cat("==============================================\n\n")
  
  results <- list()
  
  # Test CPU SketchBoost
  cat("1. Testing CPU SketchBoost (should use Py-Boost)...\n")
  tryCatch({
    model_cpu <- sketchboost$SketchBoost(
      sketch_dim = as.integer(sketch_dim),
      sketch_method = "proj",
      n_estimators = as.integer(50),
      max_depth = as.integer(4),
      learning_rate = 0.1,
      random_state = as.integer(42),
      force_fallback = FALSE
    )
    
    model_cpu$fit(as.matrix(X_train), as.matrix(y_train))
    y_pred_cpu <- model_cpu$predict(as.matrix(X_test))
    
    mse_cpu <- mean((y_test - y_pred_cpu)^2)
    r2_cpu <- 1 - sum((y_test - y_pred_cpu)^2) / sum((y_test - mean(y_test))^2)
    
    cat(sprintf("   MSE: %.6f, R²: %.6f\n", mse_cpu, r2_cpu))
    
    results$cpu_sketchboost <- list(mse = mse_cpu, r2 = r2_cpu)
    
  }, error = function(e) {
    cat(sprintf("   Error: %s\n", e$message))
    results$cpu_sketchboost <- list(error = e$message)
  })
  
  # Test XGBoost Fallback
  cat("\n2. Testing XGBoost Fallback (forced fallback)...\n")
  tryCatch({
    model_xgb <- sketchboost$SketchBoost(
      sketch_dim = as.integer(sketch_dim),
      sketch_method = "random",
      n_estimators = as.integer(50),
      max_depth = as.integer(4),
      learning_rate = 0.1,
      random_state = as.integer(42),
      force_fallback = TRUE  # Force XGBoost fallback
    )
    
    model_xgb$fit(as.matrix(X_train), as.matrix(y_train))
    y_pred_xgb <- model_xgb$predict(as.matrix(X_test))
    
    mse_xgb <- mean((y_test - y_pred_xgb)^2)
    r2_xgb <- 1 - sum((y_test - y_pred_xgb)^2) / sum((y_test - mean(y_test))^2)
    
    cat(sprintf("   MSE: %.6f, R²: %.6f\n", mse_xgb, r2_xgb))
    
    results$xgboost_fallback <- list(mse = mse_xgb, r2 = r2_xgb)
    
  }, error = function(e) {
    cat(sprintf("   Error: %s\n", e$message))
    results$xgboost_fallback <- list(error = e$message)
  })
  
  # Comparison
  if (!is.null(results$cpu_sketchboost$r2) && !is.null(results$xgboost_fallback$r2)) {
    cat("\n=== Comparison ===\n")
    if (results$cpu_sketchboost$r2 > results$xgboost_fallback$r2) {
      cat("🏆 CPU SketchBoost performs better!\n")
    } else {
      cat("🏆 XGBoost fallback performs better!\n")
    }
    cat(sprintf("Performance difference: %.6f R² points\n", 
                abs(results$cpu_sketchboost$r2 - results$xgboost_fallback$r2)))
  }
  
  return(results)
}

# Example usage (uncomment to run):
# 
# # Quick demo
# results <- demo_sketchboost(n_samples = 500, sketch_dims = c(3, 5), 
#                             sketch_methods = c("proj", "rand"))
# 
# # Individual functions:
# data <- generate_laplacian_data(n_samples = 500)
# train_idx <- sample(nrow(data$X), 0.8 * nrow(data$X))
# 
# model <- fit_sketchboost(data$X[train_idx,], data$y[train_idx,],
#                          sketch_dim = 5, sketch_method = "proj")
# predictions <- predict_sketchboost(model, data$X[-train_idx,])
# 
# # Compare implementations
# comparison <- compare_implementations(
#   data$X[train_idx,], data$y[train_idx,],
#   data$X[-train_idx,], data$y[-train_idx,],
#   sketch_dim = 5
# )