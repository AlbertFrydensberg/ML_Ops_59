
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
