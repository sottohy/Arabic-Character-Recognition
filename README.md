# Arabic-character-recognition-models

#### Expirementing with different Machine Learning classification models to see which one gives the highest accuracy. The models used are SVM, KNN, ANN, and CNN.

The dataset was obtained from kaggla: https://www.kaggle.com/datasets/mloey1/ahcd1

## Preprocessing
The data was read using Pandas and the class names were defined. Normalization was performed, then the data was explored through figuring out the number of unique classes and the number of samples in each class.

A display_images() function was defined to display 10 images when called.

## SVM
The SVM model was trained using a radial basis function (RBF) kernel on train_images and train_labels. Predictions are then made on test_images, and a confusion matrix is calculated to evaluate the model's performance. The F1 score is also calculated, then the display_images() function is called to display some test images along with their predicted and true labels. 
The F1 score was 0.56.

![svm cm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/128b8c18-d611-485d-8260-440edfb7772b)
![svm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/648146c6-0767-4e8c-9819-8e83ba648832)


## KNN
The data was split into training and validation then we define a list of k values to iterate over. For each k value, a KNN model is trained on ImgTraining and LabelTraining, and predictions are made on ImgValidation. The accuracy and F1 score are calculated for each k value, then he average F1 score is plotted against the k values to identify the optimal k value. 

![knn graph](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/2c5c51fa-9dd6-43f9-9136-3130d12d300a)

The model is tested using the best k value (8), and the display_images() function is called to display some test images along with their predicted and true labels.

![knn cm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/828f2345-8a33-478b-bf79-fba1a16eed01)
![knn](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/cf69c56e-b550-40db-911d-1238199e095d)






