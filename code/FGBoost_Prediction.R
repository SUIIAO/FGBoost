# Prediction function for a tree with depth greater than 1
predict_tree <- function(tree, X, optns = list()) {
  if (!is.null(tree$predict)) {
    
    if(optns$type %in% c("compositional", "measure", "laplacian")){
      return(lapply(1:nrow(X), function(i){
        tree$predict
      }))
    }else{
      return(rep(tree$predict, nrow(X)))
    }
  }
  
  left_indices = X[, tree$j] <= tree$threshold
  right_indices = X[, tree$j] > tree$threshold
  
  
  if(optns$type %in% c("compositional", "measure","laplacian")){
    predictions = lapply(1:nrow(X), function(i){
      return(list())
    })
    predictions[left_indices] = predict_tree(tree$left, X[left_indices, , drop = FALSE], optns)
    predictions[right_indices] = predict_tree(tree$right, X[right_indices, , drop = FALSE], optns)
  }else{
    predictions = rep(NA, nrow(X))
    predictions[left_indices] = predict_tree(tree$left, X[left_indices, , drop = FALSE], optns)
    predictions[right_indices] = predict_tree(tree$right, X[right_indices, , drop = FALSE], optns)
  }
  
  
  return(predictions)
}


# Function to predict using the trained gradient boosting model
predict_boosted_model <- function(trees, F_0, X_new, learning_rate, optns = list()) {
  # Initial prediction (same as F_0, the mean of y during training)
  if(optns$type %in% c("compositional" ,"measure", "laplacian")){
    F_new = lapply(1:nrow(X_new), function(i){
      return(F_0)
    })
  }else{
    F_new = rep(F_0, nrow(X_new))  # Initial predictions for new data
  }
  
  # Loop through each tree and update the predictions
  for (tree in trees) {
    tree_pred = predict_tree(tree, X_new, optns)
    
    F_new = lapply(1:length(F_new), function(i){
      return(pt(alpha = tree_pred[[i]]$start, 
                beta = tree_pred[[i]]$end,
                omega = F_new[[i]], 
                learning_rate = learning_rate,
                optns = optns))
    })
  }
  return(F_new)  # Final predictions
}

# Predict normally with all features considered (Tree SHAP)
predict_boosted_model_shap = function(trees, F_0, X, exclude_feature, learning_rate, optns) {
  # Initial prediction (same as F_0, the mean of y during training)
  if(optns$type %in% c("compositional", "measure", "laplacian")){
    F_new = lapply(1:nrow(X), function(i){
      return(F_0)
    })
  }else{
    F_new = rep(F_0, nrow(X))  # Initial predictions for new data
  }
  
  # Loop through each tree and update the predictions
  for (tree in trees) {
    tree_pred = predict_tree_shap(tree, X, exclude_feature, optns)
    
    F_new = lapply(1:length(F_new), function(i){
      return(pt(alpha = tree_pred[[i]]$start, 
                beta = tree_pred[[i]]$end,
                omega = F_new[[i]], 
                learning_rate = learning_rate,
                optns = optns))
    })
  }
  return(F_new)  # Final predictions
}

# Predict with a feature excluded (Tree SHAP)
predict_tree_shap = function(tree, X, exclude_feature, optns = list()) {
  if (!is.null(tree$predict)) {
    
    if(optns$type %in% c("compositional", "measure", "laplacian")){
      return(lapply(1:nrow(X), function(i){
        tree$predict
      }))
    }else{
      return(rep(tree$predict, nrow(X)))
    }
  }

  left_indices = X[, tree$j] <= tree$threshold
  right_indices = X[, tree$j] > tree$threshold
  
  # If the tree splits on the excluded feature, we average the two branches
  if (tree$j %in% exclude_feature) {
    left_predictions = predict_tree_shap(tree$left, X, exclude_feature, optns)
    right_predictions = predict_tree_shap(tree$right, X, exclude_feature, optns)
    
    # Compute the proportion of samples that go to each side
    left_weight = tree$left$num/tree$num
    right_weight = tree$right$num/tree$num
    
    predictions = lapply(1:nrow(X), function(i){
      l = list()
      l$start = left_predictions[[i]]$start * left_weight + right_predictions[[i]]$start * right_weight
      l$end = left_predictions[[i]]$end * left_weight + right_predictions[[i]]$end * right_weight
      return(l)
    })
  }else{
    predictions = lapply(1:nrow(X), function(i){
      return(list())
    })
    if(sum(left_indices)>0){
      predictions[left_indices] = predict_tree_shap(tree = tree$left, X = X[left_indices, , drop = FALSE], exclude_feature, optns)
    }
    if(sum(right_indices)>0){
      predictions[right_indices] = predict_tree_shap(tree = tree$right, X = X[right_indices, , drop = FALSE], exclude_feature, optns)
    }
  }

  return(predictions)
}
