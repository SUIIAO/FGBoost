# Fréchet Geodesic Boosting

The code for the paper 'Fréchet Geodesic Boosting'.

### Supporting software requirements

R version 4.4.0

### Libraries and dependencies used by the code

R packages to run Fréchet Geodesic Boosting:

* dplyr v 1.1.4
* tidyverse v 2.0.0
* frechet v 1.1.4
* fdadensity v 0.1.2
* Matrix v 1.7-0
* geigen v 2.3
* osqp v 0.6.3.3
* MASS v 7.3-60.2
* matrixcalc v 1.0-6
* shapes 1.2.7
* trust v 0.1-8
* pracma v 2.4.4
* purrr v 1.0.2
* manifold v 0.1.1
* foreach v 1.5.2
* doSNOW v 1.0.20
* parallel v 4.4.0
* stats v 4.4.0
* ggplot2 v 3.5.1
* shapviz v 0.9.4
* xgboost v 1.7.8.1

### Folder Structure

* `./code` code for all functions used in the paper.
* `./Code_RFWLFR` R functions to run Random Forest for non-Euclidean responses.
* `./DR4FrechetReg` R functions to run Sufficient Dimension Reduction for non-Euclidean responses.
* `./Network-Regression-with-Graph-Laplacians` R functions to run Global Fréchet Regression for network data.
* `./Single-Index-Frechet` R functions to run Single Index Fréchet Regression.
* `./Wasserstein-regression-with-empirical-measures-main` R functions to run Global Fréchet Regression for distributional data.
* `./XGBoost` R functions to run XGBoost for network data.
* `./SketchBoost` R functions to run SketchBoost for network data.
* `./Data_Application/Mortality/` code to reproduce data analysis for human mortality data in Section 6.1.
* `./Data_Application/Taxi/` code to reproduce data analysis for New York yellow taxi network data in Section 6.2.
* `./Data_Application/NJUI/` code to reproduce data analysis for emotional well-being data in Section A.2 of the appendix.
* `./Data_Application/NHANES/` code to reproduce data analysis for National Health and Nutrition Examination Survey in Section K of the appendix.
* `./Simulation/simu_dist.R` code to reproduce simulations in Section 5 for distributional data
* `./Simulation/simu_laplacian.R` code to reproduce simulations in Section 5 for network data.
* `./Simulation/simu_laplacian_xgboost.R` code to reproduce simulations in Section H for network data.
* `./Simulation/simu_laplacian_sketch.R` code to reproduce simulations in Section H for network data.
* `./Simulation/simu_comp.R` code to reproduce simulations in Section A.1 of the appendix for compositional data.

