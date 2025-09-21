library(xgboost)

train_single_dim <- function(X, y_dim, n_estimators = 100, 
                             learning_rate = 0.05, 
                             max_depth = 3,
                             early_stopping_rounds = 10) {
  tryCatch({
    # Convert inputs to numeric matrix
    X_numeric <- matrix(as.numeric(X), nrow = nrow(X))
    y_numeric <- as.numeric(y_dim)
    
    # Split into training and validation sets
    set.seed(123)
    val_index <- sample(1:length(y_numeric), size = floor(0.2 * length(y_numeric)))
    
    # Create training and validation matrices
    dtrain <- xgb.DMatrix(
      data = X_numeric[-val_index, ], 
      label = y_numeric[-val_index]
    )
    dval <- xgb.DMatrix(
      data = X_numeric[val_index, ], 
      label = y_numeric[val_index]
    )
    
    # Set parameters
    params <- list(
      objective = "reg:squarederror",
      eta = learning_rate,
      max_depth = max_depth,
      nthread = 1
    )
    
    # Train model with watchlist for early stopping
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = n_estimators,
      watchlist = list(train = dtrain, val = dval),
      early_stopping_rounds = early_stopping_rounds,
      verbose = 1  # Detailed output
    )
    
    return(model)
  }, error = function(e) {
    cat("Error in training model:\n")
    print(e)
    cat("Details of problematic dimension:\n")
    print(summary(y_dim))
    return(NULL)
  })
}

# Train models for all dimensions
train_xgboost_multidim <- function(X, y_upper, n_estimators = 100, 
                                   learning_rate = 0.05, 
                                   max_depth = 3,
                                   early_stopping_rounds = 10) {
  # Train models for each dimension
  models <- lapply(1:ncol(y_upper), function(j) {
    cat("Training model for dimension", j, "\n")
    train_single_dim(
      X, 
      y_upper[, j], 
      n_estimators = n_estimators,
      learning_rate = learning_rate,
      max_depth = max_depth,
      early_stopping_rounds = early_stopping_rounds
    )
  })
  
  # Remove any NULL models
  models <- models[!sapply(models, is.null)]
  
  # Check if any models were trained
  if(length(models) == 0) {
    stop("No models could be trained. Check your input data.")
  }
  
  return(models)
}

# Prediction function
predict_xgboost_multidim <- function(models, X_new) {
  # Verify models is not empty
  if(length(models) == 0) {
    stop("No models available for prediction")
  }
  
  # Get number of dimensions from models
  n_dims <- length(models)
  
  # Predict for each dimension
  predictions <- matrix(NA, nrow = nrow(X_new), ncol = n_dims)
  
  for(j in 1:n_dims) {
    predictions[, j] <- predict(models[[j]], xgb.DMatrix(data = X_new))
  }
  
  return(predictions)
}