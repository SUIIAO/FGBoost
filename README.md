# Fréchet Geodesic Boosting

This repository contains codes necessary to replicate **Zhou, Iao and Müller (2025)**: “Fréchet Geodesic Boosting”. The `FGBoost` functions in the `code` folder, short for Fréchet Geodesic Boosting, are designed for modeling the relationship between multivariate predictors and metric space-valued responses.

## Folder Structure

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


## Report Errors

To report errors, please contact <siao@ucdavis.edu>. Comments and suggestions are welcome.

## Citation

The Full paper can be found in "[Fréchet Geodesic Boosting](https://neurips.cc/virtual/2025/poster/118012)".

```         
@inproceedings{zhou2025fgboost,
  title={Fr{\'e}chet Geodesic Boosting},
  author={Zhou, Yidong and Iao, Su I and M{\"u}ller, Hans-Georg},
  booktitle={Advances in Neural Information Processing Systems},
  volume={},
  year={2025},
  note={in press}
}
```
