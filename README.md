# TILScout Repository

This guide will help you set up and run the TILScout analysis tool on your system. Please follow the steps below to ensure a smooth setup and execution.

## Prerequisites

Before you begin, make sure you have Python installed on your system. You will also need to install specific Python libraries required for running the TILScout.py script. These libraries are specified within the script itself.

## Installation

1. **Download the Code and Model**
   - Clone this repository to your local machine or download the zip file.
   - Ensure that you also download the `best_InceptionResNetV2_model.h5` file and place it in the same directory as the TILScout.py script. This H5 file contains the pre-trained model necessary for the analysis.

2. **Prepare Your Data**
   - Within the directory where you have the TILScout.py script, create a new subdirectory named `WSI_example`.
   - Place your whole slide image (WSI) files into the `WSI_example` directory. These are the files that will be analyzed by the script.

3. **Install Required Python Libraries**
   -openslide (version 1.2.0), tensorflow (version 2.10.0), scikit-learn (version 1.2.1), pandas (version 1.4.4), matplotlib (version 3.7.0), and numpy (version 1.23.5).
   -Other libraries required for running the TILScout.py script.
 
## Running the Script

Once you have set up your environment, you are ready to run the TILScout.py script. As the script runs, it will generate intermediate folders and files, including patches processed from the WSI and the model's prediction results. Before running the code, please ensure that the installed library versions match those listed above

## Output

The output from running TILScout.py will include directories and files containing the processed patches and the prediction results. These will be stored in the directory where the script was run, allowing for easy access and review.


