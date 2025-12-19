# WGU Data Analytics Capstone

**Comparing Classification Models: Applying McNemar’s Test to Support Data Driven Model Selection**

## Author Information

- **Author:** Dallas Taylor
- **Email:** [dtay418@wgu.edu](dtay418@wgu.edu)
- **Department:** Data Analytics, College of Information Technology

## Disclaimer

This project has been made available to showcase the `analysis.ipynb` notebook to complete my Capstone project at Western Governors University. This project may not be copied for use as any academic project, but may be used as inspiration.

Please feel free to contact me with any questions regarding its contents.

## Project Summary

When multiple classification models are trained on the same dataset, how can a machine learning practitioner demonstrate that the performance differences between models, where they exist, are statistically significant?

This report describes a training and evaluation method that can be used to directly compare any number of classification models using McNemar’s test, a special case $χ^2$ test. When paired with direct comparison of performance metrics, the results from McNemar’s test can aid machine learning practitioners in determining whether performance metrics should be the most important consideration in model selection.

## Development Environment

**Operating System:** Windows 11, 10.0.26200

**Code Editor:** Visual Studio Code, 1.107.0

**Language:** Python, 3.12.6; Pip 25.2

**Core Packages**
* Python 3 Standard Library 
    * `itertools`
    * `os`
    * `pathlib`
    * `re`
* `ipykernel`, 7.1.0
* `jupyter`, 1.1.1
* `matplotlib`, 3.10.7
* `mlxtend`, 0.23.4
* `numpy`, 2.3.5
* `pandas`, 2.3.3
* `pyarrow`, 22.0.0
* `scikit-learn`, 1.7.2
* `scipy`, 1.16.3
* `seaborn`, 0.13.2
* `statsmodels`, 0.14.6
* `tqdm`, 4.67.1

## Training Dataset

[Realistic Loan Approval Dataset | US & Canada](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada) by [Parth Patel](https://www.kaggle.com/parthpatel2130).

## Key References

1. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197

2. El Amine El Alami, S., Mouiha, A., & El Hilali Alaoui, A. (2025). Machine Learning and Deep Learning in Computational Finance: A Systematic Review.

3. Haque, F. M. A., & Hassan, Md. M. (2024). Bank Loan Prediction Using Machine Learning Techniques. American Journal of Industrial and Business Management, 14(12), 1690–1711. https://doi.org/10.4236/ajibm.2024.1412085

4. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning, with Applications in Python (1st ed.). Springer Books. https://doi.org/10.1007/978-3-031-38747-0_1

5. Patel, P. (2025). Realistic Loan Approval Dataset | US & Canada [Data set]. Kaggle. https://doi.org/10.34740/kaggle/dsv/13573669

6. Perktold, J., Seabold, S., Taylor, J., & statsmodels-developers. (2025, December 5). statsmodels.stats.contingency_tables.mcnemar. Statsmodels 0.14.6 Documentation. https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html

7. Raschka, S. (2023). mcnemar: McNemar’s test for classifier comparisons. Mlxtend Documentation. https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/

8. scikit-learn developers. (2025a). 1.1. Linear Models - Scikit Learn User Guide. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

9. scikit-learn developers. (2025b). 1.4. Support Vector Machines - Scikit Learn User Guide. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/svm.html

10. scikit-learn developers. (2025c). 1.9. Naive Bayes - Scikit Learn User Guide. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/naive_bayes.html

11. scikit-learn developers. (2025d). 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking - Scikit Learn User Guide. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/ensemble.html

12. scikit-learn developers. (2025e). 3.2. Tuning the hyper-parameters of an estimator. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/grid_search.html

13. scikit-learn developers. (2025f). KFold. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

14. scikit-learn developers. (2025g). Nested versus non-nested cross-validation. Scikit-Learn.Org. https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

15. scikit-learn developers. (2025h). RandomizedSearchCV. Scikit-Learn.Org. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

16. Walters, W. P. (2022). Comparing classification models—a practical tutorial. Journal of Computer-Aided Molecular Design, 36(5), 381–389. https://doi.org/10.1007/s10822-021-00417-2