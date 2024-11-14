# README

## Repository Workflow

This repository contains code organized in a series of numbered folders. Follow the steps below to run the workflow in numerical order in the Directory code:
1. **Folder 1: Data Preprocessing**
    - Unzip the data using `code/1_preprocessing/0_unzip.py`.
    - To understand the raw data, use the Jupyter notebook `code/1_preprocessing/1_humboldt_data_understanding.ipynb`.
    - To build the training data for each class (BOAFAB and other frog calls), use the following Jupyter notebooks:
        - BOAFAB: `code/1_preprocessing/2_data_selection_final_to_train_BOAFAB.ipynb`
        - OTHER: `code/1_preprocessing/2_data_selection_final_to_train_OTHER.ipynb`
    - To split the training data into training, test, and validation sets, use the Jupyter notebook `code/1_preprocessing/4_data_split_to_CLASSIFIER.ipynb`

2. **Folder 2: Train Generative Model**
    - To train the Diffusion Model, run the Jupyter notebook `code/2_generative/TRAIN_A_DIFF_BOAFAB.ipynb`

3. **Folder 3: Train Classification Model (for evaluation)**
    - To train the classification model, run the Jupyter notebook `code/3_classifier/1_run_model_no_data_aug.ipynb`.
    - To evaluate the classification model, run the Jupyter notebook `code/3_classifier/2_eval_model.ipynb`.

4. **Folder 4: Systematic Selection**
    - To generate a sample of generated audios and take a sample of real audios, run the Jupyter notebook `code/4_systematic_selection/1_generate_samples.ipynb`.
    - To run the systematic selection algorithm, run the Jupyter notebook `code/4_systematic_selection/2_final_figures_embedding_classifier_nodataaug.ipynb`

5. **Folder 5: Figures**
    - To create paper figures, run the notebook `code/figures/figures.ipynb`.

## Additional Information

- Ensure all dependencies are installed before running the scripts.
- Refer to the individual folder's README files for more detailed instructions.
