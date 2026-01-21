````markdown
# ml_ops_59

# ML_Ops_59

## Dataset
For this project, we will work with the **Wine dataset** available on **Kaggle**
(https://www.kaggle.com/datasets/tawfikelmetwally/wine-dataset).
This dataset is based on the results of a chemical analysis of wines grown in the same region but derived from different grape cultivars. It contains 13 analysis features of wines along with labels representing **three distinct wine classes**. The dataset is well-suited for supervised classification tasks and is small enough to allow for rapid experimentation, which is nice in our MLOps setting, as we do not seek to expand our ML skills, but rather how to work and collaborate effectively.

## Model Selection
We plan to build a supervised machine learning classifier using the **K-Nearest Neighbors (KNN)** algorithm. KNN is a simple model. The model will be implemented using the **scikit-learn** package, which provides tools for preprocessing, model training, and evaluation. For exploratory data analysis and result visualization, we expect to use **matplotlib** and/or **seaborn** to better understand feature distributions, class separation, and model performance. During the course we might discover additional third party packages that can the support the project.

## Project Description

### a. Overall Goal of the Project
The primary goal of this project is to develop and evaluate a machine learning model capable of accurately classifying wines based on their chemical properties. In addition to model performance, a major objective is to design and implement an end-to-end **MLOps pipeline**. This includes data preprocessing, data and model versioning, automated training and evaluation, experiment tracking using **Weights & Biases (WandB)**, and ensuring reproducibility. The emphasis is on building a robust and repeatable workflow rather than solely maximizing classification accuracy.

### b. Data Description
The wine dataset consists of **178 samples**, each with **13 numerical features** describing chemical attributes such as alcohol content, acidity, and flavonoid concentration. The data modality is **tabular**, and each sample is labeled with one of three wine classes. The dataset does not contain any missing values, which is convinient as data cleaning is not an important learning objective in this course.

### c. Expected Models
The expected model for this project is the **K-Nearest Neighbors (KNN)** classifier. This model will demonstrate training, evaluation, experiment tracking, and deployment within an MLOps framework.

Wine classification project with a standardized MLOps structure

## Project structure

The directory structure of the project looks like this:
```txt
.
├── .github/                      # GitHub Actions & Dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml

├── data/                         # Datasets
│   ├── raw/                      # Original, immutable data
│   └── processed/                # Cleaned / feature-engineered data

├── dockerfiles/                  # Docker build definitions
│   ├── api.Dockerfile
│   └── train.Dockerfile

├── docs/                         # Project documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md

├── models/                       # Trained and serialized models

├── notebooks/                    # Jupyter notebooks (EDA, experiments)

├── outputs/                      # Generated outputs (predictions, artifacts)

├── reports/                      # Reports and visualizations
│   └── figures/                  # Generated plots and figures

├── src/                          # Source code
│   └── project_name/
│       ├── __init__.py
│       ├── api.py                # API entrypoints
│       ├── cli.py                # Command-line interface
│       ├── data.py               # Data loading & preprocessing
│       ├── evaluate.py           # Model evaluation
│       ├── model.py              # Model definitions
│       ├── sweep_entry.py        # Hyperparameter sweep entrypoint
│       ├── train.py              # Training logic
│       ├── visualize.py          # Visualization utilities
│       ├── wandb_logger.py       # Weights & Biases logging
│       │
│       └── configs/              # Configuration files
│           ├── data/
│           │   └── wine.yaml
│           ├── model/
│           │   └── knn.yaml
│           ├── training/
│           │   └── default.yaml
│           ├── wandb/
│           │   └── default.yaml
│           └── config.yaml       # Global configuration using Hydra

├── tests/                        # Unit & integration tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py

├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml                # Python project metadata & dependencies
├── README.md                     # Project overview
└── tasks.py                      # Project automation tasks



Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
