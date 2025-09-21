# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(frechet)
library(pracma)
library(dplyr)
library(frechet)
library(fdadensity)
library(tidyverse)

library(foreach)
library(doSNOW)
library(reshape2)

ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)

source("../../code/lcm.R")
source("../../code/FGBoost.R")
source("../../code/FGBoost_BuildTree.R")
source("../../code/FGBoost_Prediction.R")
source("../../code/FGBoost_SHAP.R")
source("../../code/FGBoost_SHAP_plot.R")

# Load the mortality data
mortality = readRDS("mortality.RData")

# Extract predictor matrix and density data
x_pred = mortality$pred

# Convert density data to quantile functions
quan = foreach(i = (1:nrow(x_pred)), .combine = "rbind") %do% {
  x = mortality$density[[i]]$x
  y = mortality$density[[i]]$y
  y.quantile = dens2quantile(dens = y, dSup = x)
}
n = nrow(x_pred)

set.seed(1)
learning_rate = 0.1
n_estimators = 50
max_depth = 2

y = lapply(1:nrow(quan), function(j) quan[j,])

# result = FGBoost(x_pred, y, n_estimators = n_estimators,
#                  learning_rate = learning_rate, max_depth=max_depth,
#                  optns = list(type = "measure", impurity = "MSE",
#                               min_samples_per_leaf = 10, validation_fraction = 0.1,
#                               ncores = ncores))
# saveRDS(result, file = "mortality_FGBoost_SHAP.RData")
# result = readRDS("mortality_FGBoost_SHAP.RData")
# mortality_SHAP = compute_shap_value(model = result, X = as.data.frame(x_pred),
#                                     features = colnames(x_pred), 
#                                     all_features = colnames(x_pred))
# saveRDS(mortality_SHAP, file = "mortality_SHAP.RData")

mortality_SHAP = readRDS("mortality_SHAP.RData")

# Ensure the libraries you need are loaded
library(ggplot2)
library(dplyr)
library(tidyr)

# Convert shap_values to data frame
shap_values = as.data.frame(mortality_SHAP)

names(shap_values) = c("Population Density", "Sex Ratio", "Mean Childbearing Age", "GDP",
                       "GVA", "CPI", "Unemployment Rate", "Health Expenditure", "Arable Land")
# Number of samples (assuming this matches the number of rows in shap_values)
n = nrow(shap_values)

# Calculate global feature importance (mean absolute SHAP values)
mean_abs_shap_values <- apply(shap_values, 2, function(x) mean(abs(x)))
global_feature_importance <- data.frame(
  Feature = names(mean_abs_shap_values),
  Mean_ABS_SHAP = mean_abs_shap_values
)
global_feature_importance <- global_feature_importance[order(global_feature_importance$Mean_ABS_SHAP, decreasing = TRUE), ]

# Convert SHAP values to long format for the local explanation summary plot
shap_long <- shap_values %>%
  mutate(Sample = 1:n) %>%
  gather(key = "Feature", value = "SHAP", -"Sample")

# Create a long format of feature values for coloring
x_pred_long <- as.data.frame(x_pred) %>%
  mutate(Sample = 1:n) %>%
  gather(key = "Feature", value = "Feature Value", -"Sample")

# Merge SHAP values and feature values for plotting
shap_long <- shap_long %>%
  left_join(x_pred_long, by = c("Sample", "Feature"))

# Convert 'Feature' to factor and ensure levels are set according to global feature importance
shap_long$Feature <- factor(shap_long$Feature, levels = rev(global_feature_importance$Feature))

# Global Feature Importance Plot
global_feature_importance %>%
  ggplot(aes(x = reorder(Feature, Mean_ABS_SHAP), y = Mean_ABS_SHAP)) +
  geom_bar(stat = "identity", fill = "#FF4E42") +
  coord_flip() +
  theme_bw() +
  labs(x = "Feature",
       y = "mean(SHAP value)") +
  theme(text = element_text(size = 20),
        plot.title = element_text(hjust = 0.5))
