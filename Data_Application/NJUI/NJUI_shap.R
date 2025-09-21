# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(haven)
library(plyr)
library(tidyr)
library(dplyr)

library(foreach)
library(doSNOW)
ncores = 60
cl = makeCluster(ncores)
registerDoSNOW(cl)


entry = read_dta('entry.dta')
entry = entry %>%
  select(caseid, b2_1, b2_2, b2_3, b2_4,
         b1, b3, b4, b5, b6, b7, e1, e8b, e11, e37) %>% 
  filter(!is.na(b2_1) & !is.na(b2_2) & !is.na(b2_3) & !is.na(b2_4) & !is.na(b1) & !is.na(b3) & !is.na(b4) & !is.na(b5) & !is.na(b6) & !is.na(b7) & !is.na(e1) & !is.na(e8b) & !is.na(e11) & !is.na(e37))

id = unique(entry$caseid)
n = length(id)

yM = entry %>%
  select(b2_1, b2_2, b2_3, b2_4) %>%
  mutate(b2_1 = sqrt(b2_1/100),
         b2_2 = sqrt(b2_2/100),
         b2_3 = sqrt(b2_3/100),
         b2_4 = sqrt(b2_4/100)) %>%
  as.matrix()

y = lapply(1:n, function(i){
  yM[i,]
})

x_pred = entry %>%
  select(b1, b3, b4, b5, b6, b7, e1, e8b, e11, e37)
names(x_pred) = c("Life Satisfaction", "Education", "Marital_status", 
                  "Num of Children", "Num of People (Household)", "Income", "Hours/week", 
                  "Last Job End", "Weeks Searching", "Credit Card Balance")

source("../../code/lcm.R")
source("../../code/FGBoost.R")
source("../../code/FGBoost_BuildTree.R")
source("../../code/FGBoost_Prediction.R")
source("../../code/FGBoost_SHAP.R")
source("../../code/FGBoost_SHAP_plot.R")

n = length(y)

n_estimators = 50
learning_rate = 0.05
max_depth = 3
min_samples_per_leaf = 100
validation_fraction = 0.1

# K-fold CV
x_pred_mean = as.vector(colMeans(x_pred))
x_pred_sd = as.vector(apply(x_pred, 2, sd))

x_pred = t((t(x_pred) - x_pred_mean)/x_pred_sd)  ## standardize individually

# result = gradient_boosting(x_pred, y, n_estimators,
#                            learning_rate, max_depth,
#                            optns = list(type = "compositional",
#                                         impurity = "MSE",
#                                         min_samples_per_leaf = min_samples_per_leaf,
#                                         validation_fraction = validation_fraction,
#                                         seed = 1))
# saveRDS(result, file = "NJUI_FGBoost_SHAP.RData")
# result = readRDS("NJUI_FGBoost_SHAP.RData")
# 
# NJUI_SHAP = compute_shap_value(model = result, X = as.data.frame(x_pred),
#                                features = colnames(x_pred), all_features = colnames(x_pred))
# saveRDS(NJUI_SHAP, file = "NJUI_SHAP.RData")
NJUI_SHAP = readRDS("NJUI_SHAP.RData")

# Ensure the libraries you need are loaded
library(ggplot2)
library(dplyr)
library(tidyr)

# Convert shap_values to data frame
shap_values = as.data.frame(NJUI_SHAP)
names(shap_values) = colnames(x_pred)
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
