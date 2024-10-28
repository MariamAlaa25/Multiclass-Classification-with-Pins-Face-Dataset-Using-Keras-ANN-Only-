
Objective
------------------------------------------------------------
This project aims to build a multiclass classification model using Keras to classify images of faces from a dataset of different celebrities (or individuals) using ANNs

Dataset Description
-------------------------------------------------------------
The dataset used for this task contains multiple facial images of various individuals (celebrities) for about 105 celebrities, with each class corresponding to a different person.

Steps to Build the Model
---------------------------------------------------------------
1. Data Loading and Preprocessing:

Loading Data: Use Python libraries such as os, glob, and tqdm to 
load images and handle large datasets.

Image Size and Normalization:
Resize all images to 100x100 pixels for uniformity.
Normalize pixel values to a range between 0 and 1 to facilitate model convergence.

Dataset Split: Split the dataset into an 80% training set and a 20% validation set.

Data Augmentation: Use random flips, rotations, and zooms to improve model generalization. Data augmentation is performed using Keras’ ImageDataGenerator.

Face Detection with OpenCV: Optionally, use OpenCV’s DNN module to crop images to show only the faces, which can improve model performance.

2. Building the Artificial Neural Network (ANN):

The ANN model for face classification is built using Keras' Sequential API. The architecture includes:

Input Layer: Accepts flattened 100x100 pixel images, resulting in a 10,000-dimensional input vector.

Hidden Layers:
Use two or more fully connected (dense) hidden layers with ReLU activation (e.g., layers with 512, 256, and 128 neurons).

Apply dropout layers to prevent overfitting and potentially use batch normalization.

Output Layer: Contains as many neurons as there are unique classes (individuals) in the dataset. The softmax activation function is used to output class probabilities.

3. Model Compilation and Training:

Compile the Model: The model is compiled with:

Loss Function: Sparse categorical cross-entropy.
Optimizer: Adam.

Train the Model: Train the model with the augmented training data and evaluate it on the validation set, utilizing early stopping and a learning rate scheduler if necessary.

4. Model Evaluation and Visualization
Classification Report: After training, a classification report is generated to display precision, recall, and F1-score for each class.

Confusion Matrix: A confusion matrix is plotted to show model performance across different classes.
Loss and Accuracy Curves: The training and validation loss and accuracy curves are plotted to visualize model convergence.

Steps to Run the Code
---------------------------------------------------------------
1-Download the Notebook

2-Open juptyer Notebook

3-Create a New Notebook.

4-Install Required Packages.

5-Run the code by pressing (shift+enter)


Dependencies
--------------------------------------------------------------
pip install numpy matplotlib opencv-python keras tqdm seaborn

