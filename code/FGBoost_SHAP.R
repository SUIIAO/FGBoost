generate_subsets <- function(features) {
  subsets <- list()
  for (i in 1:length(features)) {
    subsets <- c(subsets, combn(features, i, simplify = FALSE))
  }
  return(subsets)
}

# Function to compute SHAP values for one feature
compute_shap_value <- function(model, X, features, all_features) {
  subsets = list()
  # Remove the current feature from the feature set
  for(i in 1:length(features)){
    subsets[[i]] = generate_subsets(setdiff(all_features, features[[i]]))
  }
  
  n_estimator = which.min(model$err_valid)
  trees = model$trees[1:n_estimator]
  F_0 = model$F_0
  learning_rate = model$learning_rate
  optns = model$optns
  
  # Cache predictions to avoid redundant calculations
  # prediction_cache = list()
  
  SHAP = list()
  
  text = "Feature %s - subset = %d is complete\n"
  progress = function(r) {
    if(r%%100 == 0){
      cat(sprintf(text, features[i], r))
    }
  }
  opts = list(progress=progress)
  
  # Iterate through each subset
  for (i in 1:length(features)) {
    SHAP[[i]] = foreach(
      subset = subsets[[i]], 
      # .export = c("predict_boosted_model_shap", "predict_tree_shap",
      #             "trees", "F_0", "all_features", "learning_rate", "optns"),
      .combine = "+",
      .options.snow=opts) %dopar% {
        source("../../code/FGBoost.R")
        source("../../code/FGBoost_BuildTree.R")
        source("../../code/FGBoost_Prediction.R")
        
        # Create unique cache keys for subset predictions
        cache_key_with_feature = paste(sort(c(subset, features[i])), collapse = "_")
        cache_key_without_feature = paste(sort(subset), collapse = "_")
        
        pred_without_feature = predict_boosted_model_shap(
          trees, F_0, X,
          exclude_feature = which(!(all_features %in% c(subset, features[i]))), 
          learning_rate, optns
        )
        pred_with_feature = predict_boosted_model_shap(
          trees, F_0, X, 
          exclude_feature = which(!(all_features %in% c(subset))), 
          learning_rate, optns
        )
        # Compute the weight for this subset
        weight = factorial(length(subset)) * factorial(length(all_features) - length(subset) - 1) / factorial(length(all_features))
        
        # Add the weighted marginal contribution to the SHAP value
        if(optns$type == "compositional"){
          shap_value = weight * 
            sapply(1:length(pred_with_feature), function(i){
              v = sum(pred_with_feature[[i]]*pred_without_feature[[i]])
              if(v>1){
                return(acos(1))
              }else{
                return(acos(v))
              }
            })
        }else if(optns$type == 'measure'){
          shap_value = weight * 
            sapply(1:length(pred_with_feature), function(i){
              return(sqrt(mean((pred_with_feature[[i]] - pred_without_feature[[i]])^2)))
            })
        }else if(optns$type == 'laplacian'){
          shap_value = weight * 
            sapply(1:length(pred_with_feature), function(i){
              return(sqrt(sum((pred_with_feature[[i]] - pred_without_feature[[i]])^2)))
            })
        }else{
          shap_value = weight * abs(pred_with_feature - pred_without_feature)
        }
        return(shap_value)
      }
  }
  return(SHAP)
}
