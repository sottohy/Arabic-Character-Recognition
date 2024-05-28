# Arabic Character Recognition

Expirementing with multiple machine learning models for recognizing Arabic characters from images. It includes a Support Vector Machine (SVM), K-Nearest Neighbors (KNN), a simple Neural Network, and a Convolutional Neural Network (CNN). The data is preprocessed, trained, validated, and evaluated using these models to determine their performance.



## Project Overview
This project aims to recognize Arabic characters from a set of images using various machine learning models. The dataset consists of training and testing images and labels. The project includes:
1. Data loading and preprocessing.
2. Training and evaluating SVM, KNN, a simple neural network, and a CNN model.
3. Visualizing the performance of each model.


## Data Description
The dataset was obtained from kaggle: https://www.kaggle.com/datasets/mloey1/ahcd1
It contains grayscale images of Arabic characters, each of size 32x32 pixels. The dataset is divided into training and testing sets:
- csvTrainImages 13440x1024.csv: Training images.
- csvTrainLabel 13440x1.csv: Training labels.
- csvTestImages 3360x1024.csv: Testing images.
- csvTestLabel 3360x1.csv: Testing labels.


## Dependencies
The project requires the following libraries:
- pandas
- numpy
- tensorflow
- matplotlib
- scikit-learn
- seaborn


## Preprocessing
- Loading Data: The images and labels are loaded from CSV files.
- Normalization: The image data is normalized by dividing by 255 to scale pixel values to the range [0, 1].
- Class Identification: Unique classes and their distribution are identified.


## Models
### Support Vector Machine (SVM)
- Training: The SVM model is trained using a Radial Basis Function (RBF) kernel.
- Evaluation: Predictions are made on the test set, and a confusion matrix and F1 score are calculated.

![svm cm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/128b8c18-d611-485d-8260-440edfb7772b)
![svm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/648146c6-0767-4e8c-9819-8e83ba648832)


### K-Nearest Neighbors (KNN)
- Training and Validation Split: The training data is split into training and validation sets.
- Evaluation: The KNN model is evaluated for different values of K, and the best K is selected based on F1 scores and accuracy. The best model is then evaluated on the test set.

![knn graph](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/2c5c51fa-9dd6-43f9-9136-3130d12d300a)



### Neural Network
- Architecture: A simple neural network with three dense layers.
- Training: The model is trained using categorical cross-entropy loss and Adam optimizer.
- Evaluation: The model's performance is evaluated on the validation set.

![ann](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/9c11a078-fca9-4685-b04b-a2e2a93b1ac9)


### Convolutional Neural Network
- Architecture: A CNN model with multiple convolutional and pooling layers followed by dense layers.
- Training: The model is trained using categorical cross-entropy loss and Adam optimizer.
- Evaluation: The model's performance is evaluated on the validation set.

![cnn](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/f50be2cd-25f2-4cbc-9039-91719e8a49be)


