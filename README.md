# Arabic-character-recognition-models

#### Expirementing with different Machine Learning classification models to see which one gives the highest accuracy. The models used are SVM, KNN, ANN, and CNN.

The dataset was obtained from kaggle: https://www.kaggle.com/datasets/mloey1/ahcd1


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

The model is tested using the best k value (8), and the display_images() function is called to display some test images along with their predicted and true labels. The F1 score was 0.47.

![knn cm](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/828f2345-8a33-478b-bf79-fba1a16eed01)
![knn](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/cf69c56e-b550-40db-911d-1238199e095d)



## ANN
The input images are reshaped to have a single color channel. The labels are then one-hot encoded to prepare them for classification.

The neural network is defined using a sequential model with three dense layers. The first layer flattens the input images, and the subsequent layers consist of densely connected neurons with ReLU activation functions. The output layer has a softmax activation function for classification. The model is compiled using the Adam optimizer and categorical cross-entropy loss function. It is then trained for 10 epochs. The test accuracy was 73.96 %.

![ann](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/9c11a078-fca9-4685-b04b-a2e2a93b1ac9)



## CNN

The CNN model has eight layers. The first layer is a 2D convolutional layer with 32 filters of size 3x3 and ReLU activation, followed by a max-pooling layer with a 2x2 pool size, followed by a second convolutional layer with 64 filters, a second max-pooling layer, and a third convolutional layer with 128 filters. After the convolutional layers, the model flattens the output and passes it through two dense layers. The first dense layer has 384 neurons with a sigmoid activation function, and the final output layer has a softmax activation function for classification. The model is trained for 10 epochs. The test accuracy was 91.89%.
![cnn](https://github.com/sottohy/Arabic-character-recognition-models/assets/91037437/f50be2cd-25f2-4cbc-9039-91719e8a49be)




