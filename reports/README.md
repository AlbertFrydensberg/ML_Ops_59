# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

*Group 59*


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

 *s215003, s214995, s214985, s214981*

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the third-party framework SHAP. It was integrated by adding one function call after model training, which is defined in visualize.py. It uses KernelExplainer to analyze how each wine chemical property (alcohol, flavanoids, etc.) affects KNN predictions across all three wine classes. The implementation generates a ranking showing overall feature importance. This makes sense for your MLOps project because it provides model interpretability — understanding why the model classifies wines helps validate the model works correctly and identifies which chemical measurements matter most for wine classification. Other than SHAP we used scikit.learn as well as matplotlib. scikit.learn is used to build a supervised machine learning classifier using the K-Nearest Neighbors (KNN) algorithm. The approach is chosen to have a relatively simple model we can ensure works in order to keep the focus on designing and implementing an end-to-end MLOps pipeline, which is the primary objective of this project. matplotlib is used to create static visualizations and plots. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>

> Answer:

--- Dependencies in the project were managed using a Python virtual environment combined with uv as the dependency manager. All project dependencies and their exact versions are defined in a pyproject.toml file, with a corresponding lock file generated automatically by uv. This ensures that all team members use identical package versions, eliminating the meme “works on my machine” issues and improving reproducibility. To set up an exact copy of the development environment, a new team member would follow a simple and well-defined process. First, the repository is cloned from version control. Next, a virtual environment is created locally. Once inside the project directory, the command uv sync is executed. This command reads the lock file and installs all required dependencies into the virtual environment with the correct versions. After activation of the virtual environment, the project is immediately ready to run without any manual package installation. This approach provides a fast, and reproducible setup process, making onboarding straightforward and ensuring consistency across development machines and operating systems. ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

--- The project was initialized using the cookiecutter MLOps template. From the cookiecutter template we have filled out the src/ml_ops_59 package with the core machine learning logic, including modules for data loading, model definition, training, evaluation, and visualization. We have deleted the notebooks folder as this was not useful for this particular project. We deviated from the template by adding several components that were not included by default. We added a configs/ folder with Hydra configuration and sweep files. We also added a wandb/ directory and logging utilities for experiment tracking. Additional root-level scripts for dataset statistics, data drift detection, and model registry reporting were introduced---

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

--- For linting and formatting, we used Ruff, configured through the project configuration, to enforce PEP 8–compliant formatting, consistent import ordering, removal of unused imports, and detection of common errors such as undefined variables. These checks are automatically executed in our GitHub Actions workflows to ensure code quality is maintained across all contributions. We also experimented with mypy, however it was added too late in the process. These concepts are important in larger project as they help different developers understand each others code, and enshures that it is sat up in the same way. Additionally it is very valuable when you are trying to understand code written by others. While mypy was not implemented, we have tried to add decriptions where it is necessary. We also tried to add type hints, however this is not done consistenly.  ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- In total, ten tests were implemented. Six tests focus on data validation, ensuring that the dataset loads correctly, contains the expected columns, has no missing or duplicate values, and that both features and class labels are numeric and within reasonable ranges. The remaining four tests target the model and training pipeline, verifying correct KNN model creation, valid and reproducible training accuracy, and that model predictions have the correct output shape. Together, these tests validate both data integrity and model reliability. The ten tests are naturally not exhaustive but gave us a clear understanding of how to work with tests and try to predict what could go wrong in data and model implementation ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- Total code coverage is around 46%. We would not trust that our code is error free with a coverage of 100% because the testings have to be clever and well thought for it to be error free. We cannot predict every error that we run into so there will naturally be some ongoing debugging/error solving as we execute the code.  Coverage only tells us that lines were executed during tests, not that the tests contain strong assertions or validate the right behavior. It is possible to reach high coverage with weak tests that do not catch logical errors. Thus, even with high coverage we can not be sure the code is fully operatable---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- Branches and pull requests were limited used in this project, as development was done directly on the main branch- This is also because our group was working close together in the project so we could ealisy align the work being done. Using branches and pull requests could have improved version control. Branches allow features or fixes to be developed in isolation, while pull requests enable code review and safer integration, reducing the risk of errors and improving collaboration. Branches and pull request were only used when we implementing workflows that trigges when data and the model changes. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- Yes, we used DVC for managing data in our project. We configured DVC to store data in Google Cloud Storage buckets. Conceptually DVC improved our project by enabling reproducibility and efficient collaboration. It is smart in its ability to version our datasets alongside our code in Git, permitting us to track exactly which data version was used for each model training run. This can be particularly valuable when debugging model performance issues, as we could easily roll back to previous data versions to identify whether problems stemmed from code changes or data changes. Another smart aspect is that the `.dvc` files stored in Git are lightweight metadata files that point to the actual data in cloud storage, keeping our repository small while maintaining full version control. However our dataset in this case was relatively small and did not change during the project work. It is a feature we will keep in mind for future projects. ---

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- We have organized our continuous integration into multiple GitHub Actions workflows, each responsible for a specific aspect of quality assurance and automation. One workflow focuses on code quality, where we run linting and formatting checks using Ruff to ensure consistent style and catch common programming errors. Another workflow is responsible for running unit tests, where we execute the test suite using pytest. This workflow validates the correctness of core functionality such as data loading, model training, evaluation logic. We run the same checks on multiple operating systems ubuntu, windows, and macos-latest and a fixed Python version. This helps ensure that the project behaves consistently across different environments.Anexample of a triggered workflow can be seen here:https://github.com/AlbertFrydensberg/ML_Ops_59/actions/runs/21249105272
We also have an example where the workflows fails here:https://github.com/AlbertFrydensberg/ML_Ops_59/actions/runs/21245542513
in this case we can see where the error occurs and fix it.

We also included continuous integration API tests for our FastAPI application. The API tests use FastAPI’s TestClient to verify that endpoints such as /health and /predict behave as expected.

The CI pipeline is triggered on every push and pull request to the main branch, ensuring that changes are continuously validated before being merged.
 ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We configured experiments using Hydra config files (YAML config groups under src/ml_ops_59/configs/) and tracked each run with Weights & Biases (W&B). Hydra lets us compose defaults (model, data, training, wandb) and override hyperparameters from the CLI, while W&B logs configs + metrics (and supports sweeps). Example runs:

python src/ml_ops_59/train.py task=train model.n_neighbors=7 model.weights=distance model.p=2 data.seed=42
python src/ml_ops_59/train.py task=train data.test_size=0.25


wandb agent <our-sweep-id> ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- Reproducibility was ensured through a combination of Hydra configs, fixed seeds, Git version control, and W&B experiment tracking. Each run starts from a known configuration composition (e.g., model=knn, data=wine, training=default, wandb=default) and we override values explicitly from the CLI, so the exact setup is always recoverable. During training we set np.random.seed(seed) and use deterministic data splits (train_test_split(..., random_state=seed, stratify=...)) and cross-validation (StratifiedKFold(..., shuffle=True, random_state=seed)), which makes results repeatable for the same seed/hyperparameters.

To avoid losing information, we log the full hyperparameter set, fold-level metrics, aggregated CV metrics, and artifacts (e.g., confusion matrix image) to W&B ; W&B also records run history and (optionally) the code state. To reproduce a run, we copy the hyperparameters from the W&B run config and rerun:

python src/ml_ops_59/train.py task=sweep model.n_neighbors=8 model.weights=uniform model.p=2 training.cv_folds=5 data.seed=42

 ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- In W&B we tracked both hyperparameters and evaluation metrics to compare KNN variants systematically. For sweeps, each W&B run corresponds to one hyperparameter trial (e.g., K, weights, p, cv_folds, seed). Within each trial we log fold-level metrics from Stratified K-Fold cross-validation: fold_accuracy, fold_precision, fold_recall, and fold_f1. These are important because they show stability across folds—a model with high mean accuracy but large variation across folds may be unreliable.

After all folds complete, we log aggregated CV metrics: cv_accuracy_mean and cv_accuracy_std, which summarize overall performance and robustness. We also compute out-of-fold (OOF) predictions for all samples and log cv_precision_oof, cv_recall_oof, and cv_f1_oof. OOF metrics are useful because they approximate “performance on unseen data” across the full dataset without training on the same sample it is evaluated on.

Finally, we log a confusion matrix in two forms: (1) as a saved PNG image (confusion_matrix.png) and (2) as an interactive W&B confusion matrix plot. This helps diagnose which classes are confused (e.g., class 2 vs class 3), which raw accuracy alone can hide.

We have used the the sweep.yaml in the root of the project

--- ![WandB result1](figures/wandb1.png)
--- ![WandB result2](figures/wandb2.png)



link to WandB: https://wandb.ai/s214995-danmarks-tekniske-universitet-dtu/ML_Ops_59/sweeps/uhox2i0o?fbclid=IwY2xjawPgMBNleHRuA2FlbQIxMABzcnRjBmFwcF9pZBAyMjIwMzkxNzg4MjAwODkyAAEei8nIPgDCTNzoY1yEattiHPFFvaBWNMMELrGOuJq4kLJmGtrKd-r-Y97pNoU_aem_bkyW5V-q1tyQMqSD85UF0g&nw=nwusers214995

 ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

---  We developed two Docker images: one for training and one for API deployment. Both Dockerfiles use the `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base image for fast dependency management with uv, and implement cache mounting for faster rebuilds, as shown in the guide. We separated these into two distinct Dockerfiles because they serve different purposes. The training container runs batch jobs to train models and needs write access to save checkpoints. The API container runs continuously as a web service to serve predictions to users. It uses unicorn for web acess. We initially considered creating a separate evaluation Dockerfile, but since evaluation is already integrated into our training pipeline (train.py), a separate container would be redundant. This is something we'll be aware of in future cases. All Dockerfiles use the `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base image for fast dependency management with uv. To run the training docker image: `docker run --rm train:latest` Link to docker file for training: <https://github.com/AlbertFrydensberg/ML_Ops_59/blob/main/dockerfiles/train.dockerfile>  ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

---  Debugging method was dependent on group member. Some just used the continuous integration setup and found errors when commiting code to github and others ran the experiements, found errors and tried to backtrack the errors by printing statements in the code or running only parts of the script. We also did some unit testing and model testing during the experiement phase at we could not foresee all the tests required for our code before be began to run the experiments. We also ran single profiling at some point on our train.py which showed that almost everything was 0.000, and we interpreted that as - either the script did very little work, or the expensive part wasn’t executed from that entry point. But we think it could be a usefill check in the future when working with way larger datasets and training setups to see where the most extensive runtimes are and identify potential optimisation areas in our code. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

GCS was used to store datasets and experiment artefacts. The storage bucket served as a central, versioned data backend, integrated with DVC, allowing data to be shared consistently between local development and cloud-based compute resources. Compute Engine was used for model training. By creating CPU-based VMs, training could be executed on scalable cloud hardware without being limited by local resources. This allowed experiments to run for longer durations and with higher compute capacity when needed. Cloud Run was used to deploy a FastAPI application. Cloud Run provides a fully managed, serverless environment for running containerised web services, enabling the FastAPI backend to be deployed with automatic scaling and without manual infrastructure management.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to run our KNN training with the CPU-based VMs, training could be executed on scalable cloud hardware without being limited by local resources. This allowed experiments to run for longer durations and with higher compute capacity although the higher capacity is not really needed for this project it is nice to know how to manage training when scaling.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- ![Build result](figures/build.png)
The figure above shows the GCP Cloud Build history for our project. It documents multiple container builds triggered during the deployment attempts of our FastAPI service to Cloud Run. Each entry corresponds to a build initiated by gcloud run deploy, including timestamps, regions, and build durations. The build history highlights an important aspect of cloud-based development: iterative debugging and redeployment. Several builds failed due to configuration issues (we worked with Dockerfile placement, build context, and command syntax...), which is clearly reflected in the repeated failed build entries. This history is useful both for debugging – we could inspect build logs and for reproducibility, as each build is uniquely identified and traceable.---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- Yes, the model was successfully trained in the cloud using Google Compute Engine. A virtual machine was created with a standard Ubuntu image and CPU resources, which was sufficient for the KNN model used in the project. Compute Engine was chosen because it provides a simple and transparent way to scale training while maintaining full control over the execution environment. After creating the VM, it was accessed via SSH and configured in the same way as the local development environment. The project repository was cloned from GitHub, dependencies were installed using the uv package manager to ensure environment reproducibility, and the dataset was downloaded from Google Cloud Storage using DVC. Training was then executed by running the existing training script directly on the VM. ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- We did manage to write an API for our model. We used FastAPI and implemented the service in api.py with three endpoints: a /health endpoint for a simple status check, and /predict and /predict_batch for single and batch inference. The API loads the trained model artifacts (KNN model + StandardScaler + feature order) once at startup using FastAPI lifespan events, which keeps requests stateless and avoids reloading the model on every call. Inputs are validated with Pydantic schemas, and we allow clients to send either a list of feature values (in the correct order) or a dictionary mapping feature names to values. Before inference we convert inputs into a DataFrame with the expected column names, apply the saved scaler, and return predicted wine classes as strings. We also saved artifacts to a dedicated models/ folder to decouple training from inference and make later deployment easier. The API was developed and tested locally, which allowed us to focus on correctness, reproducibility, and testing rather than deployment complexity. We chose not to deploy the API to the cloud due to time constraints ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- We did manage to deploy our API locally, but we did not deploy it to the cloud. The API was served locally using FastAPI together with uvicorn as the ASGI server. After training the model and saving the artifacts, we started the service with uv run uvicorn ml_ops_59.api:app, which launched the API on http://localhost:8000. The service could then be invoked either through the automatically generated Swagger UI at /docs or programmatically using tools such as curl.

For example, a prediction could be obtained by sending a POST request to the /predict endpoint with a JSON payload:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x":[13.2,1.7,2.3,15.6,100,2.8,3.1,0.3,1.9,5.6,1.0,3.2,1000]}'

We chose not to deploy the API to the cloud due to time and scope constraints, and instead focused on ensuring that the local deployment was robust, testable, and reproducible. The API design should make it deployable in cloud.
   ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- We performed API testing using pytest and FastAPI’s TestClient, which allowed us to run integration-style tests against the endpoints without starting an external server. In test_api.py we test that /health returns HTTP 200 and a valid JSON response, and that /predict returns a valid prediction string. The predict test reads models/metadata.json to automatically match the number of expected features, ensuring the test stays consistent even if we retrain and overwrite the artifacts in models/.

For load testing we used Locust. We created a minimal locustfile that repeatedly sends POST requests to /predict with a deterministic payload (a zero-vector of the correct feature length, also read from metadata.json). Locust then reports throughput (requests/sec), failure rate, and response time statistics (average and percentiles such as p95/p99). These metrics help quantify performance and whether the API remains stable under concurrent user load.

Step-by-step guide to test:

Ensure dependencies are installed

uv sync

Train once (so models/ artifacts exist), e.g.

uv run python -m ml_ops_59.train model.n_neighbors=8

Start the API (Terminal 1)

uv run uvicorn ml_ops_59.api:app --host 127.0.0.1 --port 8000

Confirm it works (Terminal 2)

curl http://127.0.0.1:8000/health

Run Locust

uv run locust -f tests/performancetests/locustfile.py --host http://127.0.0.1:8000

Visit website and go nuts.

Below is a load test result for 50 users and 5 spawn rate:

![Load test result](figures/load_test.png)

 ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We did not fully manage to deploy the model for production in a fully correct version. However, we did set up some monitoring in our cloud for data drift monitoring of feature distributions (e.g. alcohol content, acidity, phenols) between a reference training dataset and the set used for the model. Here we used tools as Evidently, where we could detect changes in feature distributions over time. We could also have implemented  prediction monitoring (the monitoring concept and learning of it should be the same) which could track class distribution over time. In a real implementation this could be useful as a sudden shift in predicted wine classes could indicate drift, data quality issues, or changes in data collection procedures. We also implemented a little service-level monitoring for error rates in GCP Cloud Monitoring for the FastAPI service deployed on Cloud Run.---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- During the project, it was group member s214985  who used GCP credits. In total, approximately 300 DKK worth of credits were consumed over the course of development. The service accounting for almost all of this usage was Compute Engine, which was used for model training. Training was executed multiple times to test different configurations (different K values) and validate the pipeline, making Compute Engine the dominant cost driver. Despite the repeated training runs, the overall cost remained low because the dataset was relatively small and the chosen model was a K- KNN classifier. What maybe also could explain the lower about of credits is that KNN has no expensive optimisation phase compared to more complex models such as deep neural networks, and training primarily consists of storing the dataset rather than performing iterative gradient-based updates. ---


### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- One of the biggest struggles in the project was related to collaboration and version control practices. At several points, multiple group members worked directly on the same files on the main branch instead of using separate feature branches. This occasionally led to merge conflicts, some of which failed or required manual resolution. These conflicts were time-consuming and sometimes resulted in accidental overwrites or the need to reapply changes. We overcame this challenge by better communication, making sure that we were not working on the same at the same time. For future projects it would be a good idea to have a more stric branching discipline. Another major challenge in the project was integrating cloud storage using DVC. This step took a significant amount of time to implement, as understanding the underlying concepts of cloud storage buckets and configuring them correctly in Google Cloud proved difficult. In particular, setting up the remote storage, managing credentials, and ensuring that DVC could reliably push and pull data from the cloud required several iterations before it worked as expected. To be honest we just kept trying untill it worked .  ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

>--- All members of the group actively participated in the development of this project. We worked together in person on all days dedicated to project work, often solving tasks collaboratively. In several cases, multiple group members worked on the same task even though only one person ultimately pushed the changes to the repository.
Student s214985 was primarily responsible for setting up and configuring Google Cloud Platform (GCP). s214995 was in charge of setting up the API, s214981 handled Docker and containerization, and s215003 was responsible for setting up the GitHub repository and integrating continuous workflows.
We used ChatGPT to help debug errors, clarify unfamiliar concepts, improve code structure and readability, and assist with integration across different parts of the project.  ---
