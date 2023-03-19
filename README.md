
# Gold Price Preedictor

This is a Python code that demonstrates how to use TensorFlow to create a simple neural network to predict the price of gold (GLD) based on other economic factors. The code imports necessary libraries including numpy, pandas, matplotlib, seaborn, and tensorflow.

The code starts by loading a CSV file containing the data for the gold price and converting it to a parquet file format. After loading the parquet file into a Pandas DataFrame, the code separates the dataset into input features (X) and the target variable (Y).

The code then splits the dataset into training and testing sets using the train_test_split method from sklearn. It creates TensorFlow datasets using the training and testing sets, which can be used to train and test the model.

Next, the code builds a neural network model with one input layer with 64 neurons and an output layer with one neuron. The model is compiled with a mean squared error loss function and an Adam optimizer.

The model is trained using the fit method of the model object, which is passed the training dataset and the number of epochs (iterations). The trained model is then evaluated using the test dataset, and the loss is printed to the console.

Finally, the model is used to make predictions on the test dataset, and the R-squared error is calculated using the r2_score method from the sklearn.metrics module. The R-squared score is a measure of how well the model fits the data, with a score of 1 indicating a perfect fit.

Overall, this code demonstrates how to create a simple neural network model using TensorFlow to predict the price of gold based on other economic factors, and how to evaluate the model's performance using the R-squared score.


## For profiling the code

Type this command in cmd to execute and display the profiling results of the code

```cmd
python -m cProfile file_name.py
```


## Documentation

Here's the documentation for this project, read this you'll get everything! :p

This code is a machine learning model that predicts the price of gold using historical data.

The code imports necessary libraries such as NumPy, Pandas, Matplotlib, Seaborn, Tensorflow, and Scikit-learn.

It first reads in a CSV file containing historical gold price data and converts it to a Parquet file format. Then it reads in the Parquet file into a Pandas DataFrame.

Next, it separates the DataFrame into predictor (X) and target (Y) variables. The predictor variables are all columns except for the "Date" and "GLD" columns, while the target variable is the "GLD" column. The data is then split into training and testing sets using a 80-20 split and a random seed of 2.

The code then converts the training and testing data from Pandas DataFrames to TensorFlow datasets.

The machine learning model is built using a Sequential model from the Keras API within TensorFlow. It contains two Dense layers, one with 64 neurons and a ReLU activation function, and the other with a single neuron. The loss function used is mean squared error (MSE), and the optimizer used is Adam.

The model is trained on the training data using the fit method for 10 epochs.

After training, the model is evaluated on the testing data to calculate the loss using the evaluate method.

Finally, the code predicts the gold prices using the testing data and calculates the R squared error using Scikit-learn's r2_score function. The R squared error is a measure of how well the model fits the data, with a value of 1 indicating a perfect fit and a value of 0 indicating no fit at all. The closer the value is to 1, the better the model's performance.

