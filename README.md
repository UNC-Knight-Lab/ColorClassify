# ColorClassify

## Description
This Python package was developed by the Knight Lab at UNC Chapel Hill to quantify one-bead one-compound libraries stained for a colorimetric assay. It consists of two parts, an image quantification section that returns RGB values of selected beads, and a second will classify lists of RGB values into groups based on pre-classified test data.

## Installation Instructions
A version of Python >= 3.7 is required to use this package. We recommend using [Anaconda](https://www.anaconda.com) to install Python for those new to Python.
1. Open the terminal (MacOS) or Command Prompt (Windows).
2. Download the package by either:
   1. Download the zip from GitHub (Code -> Download ZIP). Unzip the package somewhere (note the extraction path). The extracted package can be deleted after installation.
   2. Clone this repository (requires git to be installed) with:
      
   `git clone https://github.com/UNC-Knight-Lab/ML-bead-analysis.git`

3. Install the package using pip. This command will install this package to your Python environment.
    The package path should be the current working directory `.` if cloned using git. Otherwise, replace it with the path to the `ML-bead-analysis` folder.
      
   `pip install .`
   or `pip install /path/to/package/ML-bead-analysis`

That's it!

## How to use
This tool can be run as a Python function or from the command line terminal. The tools will prompt for user input. Sample data is included for both image quantification and classification. 

### Image quantification
This tool is designed to determine the average RGB color of individual beads in an OBOC library. 
1. To begin, designate an input folder containing .tif images of the library. An sample image is provided in the sample data.
2. The script will display each image in successfion. For each, click the approximate center of each bead to be quantified. When finished, click X to exit the image window, and an annotated version with the region of interest over each bead will be displayed.
3. The next image will be output automatically until all the images in the library folder have been processed. Annotated images as well as the RGB values of the regions for each images are exported to new folders created within the input folder.

### RGB classification
This tool is designed to classify RGB values of a list of beads into color assignments.
1. To begin, designate an input folder containing excel files of the RGB values of beads in the first three columns and color group designated in the fourth column. A sample file is provided titled "data_train".
2. A gradient boosting classifier will be fit to the training data. An optional gridsearch optimization can be run to determine the best model hyperparameters or these can be otherwise provided.
3. Predictions with RGB values with unknown classification will be performed over a specified number of iterations. Averaged predictions will be deposited into the location specified by the input folder.

### To run from terminal:
For the image quantification, use:

    image_quant -i "/path/to/input_folder"

For the classification, use:

    bead_classification -i "/path/to/input_folder"
    
Instead of specifying an input or output folder, you can also navigate to your data input folder in the terminal and run the script.
The current working directory will be used as default.
Use the help `-h` tag to see more options.

### To run in Python:
For the image quantification, in a Python environment, import the Python function:

    from bead_quant.image_quant import image_quantification
    image_quantification(input_folder)

For the classification, in a Python environment, import the Python function:

    from bead_quant.bead_class import classification
    classification(input_folder)
