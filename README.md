# README
This is the final submission for module ELEC0134 AMLS_20/21.
Note: This is a throwaway github account, the git commit history can be provided upon request as it is
on my main account.

## How to run
To run project, upload the required datasets into the empty dataset folder, and run main.py.
The directory names are unchaged from the originally given datasets, so it should work.

## Brief Description of Organization
Folders A1, A2, B1, B2 are the classes which contain the model classifier used for the corresponding task.
Some classes, (like A1 and A2) are virtually identical, as similar models are used for task A. However, they
are seperated to follow assignment structure.

The files all have header comments to explain their purpose.

The Datasets folder is empty, and required the celeba, cartoon, celeba_test, and cartoon_set_test folders to be
uploaded to the folder for the code to function.

The UtilityA folder contains all utility associated with TaskA.
     - 'models.py' is a file with 8 basic scikitlearn models which are ran for testing
     - 'plots.py' is a file which has the functions used to make some of the plots in the report
     - 'pre_processing.py' is for pre_processing
     - 'utility.py' is adapted from the labs, and is used for feature extraction
     - 'validation.py' is used for parameter tuning and model validation

The utilityA folder contains all utility associated with TaskA.
    - The files follow a similar purpose structure as the UtilityA folder
    - 'utiltyB.py' contains the eye_extractor function for getting eye features from the cartoon dataset
    - 'eye_extraciton_test.py' is a test file which is left for illustration purposes of the eye feature extractor  

## Required packages
OpenCV, sklearn, Keras, matplotlib, dlib (and required dependencies), pandas, numpy
