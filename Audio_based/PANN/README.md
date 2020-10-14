PANN-based audio model
==============================

The PANN-based project Organization
------------

    ├── LICENSE
    ├── README.md                           <- README for developers using this project.
    │
    ├── notebooks                           <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                          the creator's initials, and a short `-` delimited description, e.g.
    │                                          `1.0-jqp-initial-data-exploration`.
    │
    ├── logs                                <- Serialized models and confusion matrices.
    │   └── tb                              <- Tensorboard files
    │
    └── src                                 <- Source code for use in this project.
        ├── __init__.py                     <- Makes src a Python module
        └── utils                           <- Additional scripts
            ├── __init__.py                 <- Makes utils a Python module
            └── notebooks_utils.py          <- Notebooks functions

--------

Note:
* All experiments were carried out in Jupiter Notebook.
* Despite the fact that a random seed is fixed, different computers will give different results.
* You should create folders logs/tb and extract features for reproduce results. In addition, you should configure the data paths.
* You should clone https://github.com/qiuqiangkong/audioset_tagging_cnn into root folder.
* You should download Cnn14_mAP=0.431.pth model.

# Reproduction of Results
Use notebook ```1.0-Maxim-WeightedModelWithExtractor.ipynb``` to reproduce the results PANN model. You should preliminary define all paths, split files into vocals and accompaniment, and set ```is_train_mode = True```.
