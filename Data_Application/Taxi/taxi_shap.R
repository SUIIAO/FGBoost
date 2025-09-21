# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(plyr)
library(tidyr)
library(dplyr)

library(foreach)
library(doSNOW)
ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)


# Load Data
x_pred = readRDS("taxi_predictor_2017_2018.RData")
y_nw = readRDS("taxi_laplacian_2017_2018.RData")
meanLapM_y = sum((Reduce("+", y_nw) / length(y_nw))^2)

source("../../code/lcm.R")
source("../../code/FGBoost.R")
source("../../code/FGBoost_BuildTree.R")
source("../../code/FGBoost_Prediction.R")
source("../../code/FGBoost_SHAP.R")
source("../../code/FGBoost_SHAP_plot.R")

n = length(y_nw)
k = 10

n_estimators = 100
learning_rate = 0.1
max_depth = 2
early_stopping_rounds = 10

x_pred_mean = as.vector(colMeans(x_pred))
x_pred_sd = as.vector(apply(x_pred, 2, sd))

x_pred_mean[which(names(x_pred) %in% c("MTWT", "FS"))] = 0
x_pred_sd[which(names(x_pred) %in% c("MTWT", "FS"))] = 1

x_pred = t((t(x_pred) - x_pred_mean)/x_pred_sd)  ## standardize individually
colnames(x_pred) = c("Mon to Thur", "Friday and Saturday",
                     "Fare Amount", "Tip Amount", "Tolls Amount",
                     "Trip Distance", "Passenger Count",
                     "Temperature", "Wind", "Precipitation", "Pressure", "Humidity")
# result = FGBoost(x_pred, y_nw, n_estimators,
#                  learning_rate, max_depth,
#                  optns = list(type = "laplacian", impurity = "MSE", min_samples_per_leaf = 10,
#                               validation_fraction = 0.1, ncores = ncores,
#                               early_stopping_rounds = early_stopping_rounds))
# saveRDS(result, file = "taxi_FGBoost_SHAP.RData")
# result = readRDS("taxi_FGBoost_SHAP.RData")
# taxi_SHAP = compute_shap_value(model = result, X = as.data.frame(x_pred),
#                                features = colnames(x_pred), 
#                                all_features = colnames(x_pred))
# 
# saveRDS(taxi_SHAP, file = "taxi_SHAP.RData")
taxi_SHAP = readRDS("taxi_SHAP.RData")

# Ensure the libraries you need are loaded
library(ggplot2)
library(dplyr)
library(tidyr)

# Convert shap_values to data frame
shap_values = as.data.frame(taxi_SHAP)
names(shap_values) = colnames(x_pred)
# Number of samples (assuming this matches the number of rows in shap_values)
n = nrow(shap_values)

# Calculate global feature importance (mean absolute SHAP values)
mean_abs_shap_values = apply(shap_values, 2, function(x) mean(abs(x)))
global_feature_importance = data.frame(
  Feature = names(mean_abs_shap_values),
  Mean_ABS_SHAP = mean_abs_shap_values
)
global_feature_importance <- global_feature_importance[order(global_feature_importance$Mean_ABS_SHAP, decreasing = TRUE), ]

# Convert SHAP values to long format for the local explanation summary plot
shap_long = shap_values %>%
  mutate(Sample = 1:n) %>%
  gather(key = "Feature", value = "SHAP", -"Sample")

# Create a long format of feature values for coloring
x_pred_long = as.data.frame(x_pred) %>%
  mutate(Sample = 1:n) %>%
  gather(key = "Feature", value = "Feature Value", -"Sample")

# Merge SHAP values and feature values for plotting
shap_long = shap_long %>%
  left_join(x_pred_long, by = c("Sample", "Feature"))

# Convert 'Feature' to factor and ensure levels are set according to global feature importance
shap_long$Feature = factor(shap_long$Feature, levels = rev(global_feature_importance$Feature))

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

