---
title: "Understanding Neural Networks with Iris Dataset"
date: 2023-01-08
lastmod: 2023-08-30
author: "Lewis Watson"
authorLink: "https://lnwatson.co.uk"
description: "Intro look at MLP Neural Networks"
draft: false

tags: [Beginner, AI/ML, Machine Learning]
categories: [Machine Learning]

hiddenFromHomePage: false
hiddenFromSearch: false

math: true
---

> Future edits may not appear on this github but will be live [here](https://lnwatson.co.uk/posts/intro_to_nn/).

# Understanding Neural Networks with Iris Dataset

Today we will specifically look at implementing solutions to classification problems. A classification problem is where we are trying to categorise our input into two or more discrete categories. For example we could be categorising hand drawn numbers, that's A-Z, a-z, 0-9 - This would be a multiclass classification problem with 62 classes. On the other hand, if we are classifying between malignant skin marks vs benign - this would be a binary classification problem.

## Problem Features and Goals
To help get a feel for how a simple neural network would work, lets use the following example: we want to classify which of three species of iris (flower) we are looking at. In order to distinguish between them we will use four measurements.

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width


In machine learning, these are what we call features. Features can be extracted in a number of ways, for example in the case of classifying positive or negative product reviews, we might decide to count the number of specific tone-related words in a sentence (i.e excellent, terrible, shocking).

In this example, we are making the assumption that the measurements are accurate and that we will receive them in numerical form.

Note: Often these features are much more complicated and can also extracted by the model itself, for example in image classification. I will talk about image classification with CNNs and RNNs in a different post.

The goal of the model is to be able to accurately classify from these features the species of iris. There are three species we will train this model to identify: Iris Setosa, Iris Versicolor, and Iris Virginica.


## A Step Back - What even is a neural network

Before we start looking at code, lets take a step back and outline the structure of a fully connected neural network (also known as a multilayer perceptron or MLP). Put simply, there are three 'sections'. We have the input layer, our hidden layer(s), and finally the output layer.

![MultiLayer Perceptron Diagram](https://miro.medium.com/max/828/1*-IPQlOd46dlsutIbUq1Zcw.webp)

[Image Source: becominghuman.ai/](https://becominghuman.ai/multi-layer-perceptron-mlp-models-on-real-world-banking-data-f6dd3d7e998f)

### Neurons
**A neuron**, is a unit that takes inputs and produces an output. In a neural network, neurons are arranged in layers, and our input is sent through the neurons in the input layer, to the neurons in the hidden layer (there can be multiple hidden layers), and finally to the neurons in the output layer.

> It is important to note that when we talk about the 'input' into a layer, we are talking about the output from the previous layer, or from the data we put into the input layer.

It can be useful to think of a neuron as an object, like in oop, where each neuron has properties/functions such as inputs, weights, bias, activation function, and output. Each neuron is connected to all the other neurons in the next layer, and the 'weights' or the strength of these connections form the weight matrix.

### Neuron Activation
The output of a neuron is calculated by applying the "activation function" to the weighted sum of inputs and bias. I will go more into details about activation functions at a later date, but for now just know they are the *functions used to decide the output of a neuron.*

A Neuron also has a bias, this is a value added to the weighted sum of inputs of a neuron, and is used to shift the activation function (f) to the left or right.

Finally, the output of a neuron is the result of the activation function being applied to the weighted sum of inputs and bias.

### Network
The whole network is just a collection of neurons connected together in layers. The input layer takes the data we are feeding into the network, and each successive layer is the output of the previous layer, until we reach the output layer.

Once we have the output of the output layer, we can interpret it to get the result of the network. In the case of a classification task, these outputs can be interpreted as probabilities of the data belonging to each class.

---

Okay hopefully that didn't scare you away. I promise it'll make sense, lets now look at how a neural network 'learns'.


# Training a neural network

In this example we will only look at supervised learning. This is when we have the model ‘learn’ aka train using known correct input output pairs. Learning can be though of as an optimisation problem, aiming to map these sample pairs. So what are the steps?

## Outline - Stages of training

### Pre-processing
This is when we prepare the data for the network, converting it into a format that our neural network will understand. During this stage we also can do a number of other things such as normalising, scaling, and even splitting the data into sub-datasets for training, validation, and testing once training is complete.

### Model Selection/Definition
Now we define how we want our network's structure to look. In the Iris example problem, we have three species and four input features.
This means that we have to start with an input layer that has four neurons (one for each feature), and an output layer with three neurons (one for each class of species).

In-between, we have the hidden layers. There can be an endless number of them, the more layers a network has the 'deeper' it is. This is where the term Deep Neural Network comes from - this will also be a topic at a later date.

### Training (Scary!)
The training stage is when we adjust the weights and biases for the neurons in our network, this ‘adjusting’ of the strengths of the features connections to each other neuron. This optimisation of the weights and biases is done using an algorithm such as gradient descent. Generally speaking, we are optimising the weights and biases such that our training data’s inputs (the features) match their outputs (labels).

### Evaluation, Hyperparameter optimisation, and Re-Training!
The evaluation stage is when we measure the performance of our network against unseen data. If the network is performing well, we can go back and try to improve the performance by tweaking the hyperparameters. These hyperparameters are the settings of the network such as the learning rate (how fast we want to make changes), the number of hidden layers, and the number of neurons in each layer. Once these hyperparameters have been adjusted, we can then go back to the training stage and re-train the network.

### Deployment
Finally, when the network is performing satisfactorily, it is time to deploy it. This can be done by exporting the model weights and biases and using them in a production-ready system. This system can then be used to predict the output of unseen data, or even to control a physical system such as a robot or an autonomous car.


## Let's build a MLP

Okay, enough talk, lets code! More specifically lets write and train a multiclass iris classification multilayer perceptron neural network.
If you would like to code along [see here for the IRIS.csv dataset used.](https://github.com/lewiswatson55/Understanding_Neural_Networks_with_Iris_Dataset/blob/master/IRIS.csv) 

This tutorial will assume you know how to set up a python environment, and also install libraries, we will use: Pandas, TensorFlow, and sk-learn

### Import our libraries and get set up

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
```

Pandas is a pretty standard library to use in machine learning and data science for data manipulation. We are also using scikit-learn's handy train_test_split function to make our lives easier and focus on understanding neural networks. And finally, our model will be written using Keras from tensorflow - this is a framework that makes it easy to build deep learning models.


### Preprocessing time

Most importantly, we need to load in our IRIS.csv dataset. Out of interest, the dataset we are using is public domain and can be found [here](https://www.kaggle.com/datasets/uciml/iris)

However, the version I'm using has been slightly modified and can be [downloaded from my github here](https://github.com/lewiswatson55/Understanding_Neural_Networks_with_Iris_Dataset/blob/master/IRIS.csv). Alternatively, uncomment the last line in the code provided below.

### Loading our dataset into a dataframe

```python
# Load our csv into a dataframe with column names
df = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# extract the last column as the labels
labels = df.pop('species')
# df.pop('Id') # Uncomment this line if you downloaded the iris dataset directly from kaggle
```

Okay first we are using pandas to read the csv dataset and we are also telling it the name of the columns.
Then we use the pop method to extract the species column into its own dataframe.

Optionally, we might need to remove the 'Id' column if the dataset was downloaded directly from kaggle or somewhere else - we want to ensure our dataframe (df in example code is) containing our inputs and nothing else.

Before we move on lets quickly discuss one-hot encoding.

### One-Hot Encoding

Instead of using a label or id to identify each kind of species, we can use one-hot encoding. Lets pretend we want to represent the labels: Apple, Orange, and Banana. To do this we will use three bits, one for each discrete label or category.

|Label|apple|orange|banana|
|-----|-----|------|------|
|apple|1    |0     |0     |
|orange|0    |1     |0     |
|banana|0    |0     |1     |

As we can see, each row has a binary representation of either 0 or 1. This is a useful way of representing labels or categories in a way that a machine or computer can understand.

Thankfully, pandas has a handy function to create this one-hot encoding for us.

```python
# convert the labels to one-hot encoding
labels = pd.get_dummies(labels)

```

### From dataframes to numpy arrays

Dataframes are great but for our model we will need to convert them into a numpy array as this is what keras will expect as input. Now lets convert the features and labels.

```python
# convert the dataframes to numpy arrays
X = df.values
y = labels.values
```

Simple right? Now that's done we are almost ready to set up our network!
One quick note: notice we've used X and y - this is the standard way to represent inputs (features) and outputs (categories).

### Training vs Validation vs Test Data

In machine learning it is best practice to use both validation and test data alongside our standard training data. However, it is important to ensure that there is no overlap between these. The majority of our data should be in the training set, often validation takes 20% of the data and test even less - I will be using 5% for test.

So what is the difference between training, validation, and test data? Training data is the data actually used to update the weights in our network whereas, the validation data is used to help verify that our model is not overfitting. Overfitting is when the model accurately classifies the data we used to train the model but doesn't generalise when new unseen data is evaluated. Validation testing is checked multiple times (at each epoch/run) to see if the model is overfitting. Finally, test data is used at the very end for a second check to ensure the model is not overfitting and will generalise.

To create our 20% validation, and 5% test data we can use the scikit-learn `train_test_split()` function.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
```

Perfect, now we have X_train, X_val, X_test as well as their associated labels (y).


### Model Selection/Definition

```python
# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(4,)))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()
```

For our model we will start using Keras' Sequential method. This method allows you to build a linear stack of layers, with the output of one layer being the input of the next. The output of the final layer is the output of the model.

So we already know our input layer is of size four (each feature), now we can design the hidden layers. In keras the Dense layer type is a fully-connected layer with n number of neurons. For the first hidden layer, I have chosen 16 neurons. This gives us a layer total of `16+(16*4)` or 80 trainable weights and biases. That is 64 weights `4*16` to each neuron. Plus 16 biases for each neuron.

The second hidden layer uses three neurons, this will be our output layer for the model. Giving us an additional `3+(16*3)` or 51 trainable parameters for a total of 131 trainable parameters in the network.

We also use an activation function, which we recall from earlier where we said is the function that takes the sum of the weights and bias to give our output. Well the two different activation functions used in this model are used to create different types of outputs depending on the type of layer. The first activation function, ReLU (rectified linear unit), is used in the first layer to create a non-linear output, which helps the model to better learn complex patterns in the data. The second activation function, softmax, is used in the second layer to create a probability output, which allows the model to assign a probability to each of the possible classes.


### Compile the model!

```python
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

The next step is to compile our model. This is done by specifying the optimizer, loss function, and metrics that we want to use.

The optimizer controls the learning rate of the model and is used to update the model's weights and biases. We have chosen to use the Adam optimizer, which is a popular choice for many deep learning tasks.

The loss function measures the difference between the predicted and actual values and is used to determine how well the model is performing. In this case, we are using the categorical cross-entropy loss function, which is used for multi-class classification tasks.

Finally, we specify the metric that we want to use. In this case, we are using accuracy, which measures how often the model correctly predicts the correct class.


### Training time (less scary now)

```python
# Train the model!
model.fit(X_train, y_train,
          batch_size=12,
          epochs=200,
          validation_data=(X_val, y_val))
```

Now it's time to train the model. We do this by passing in the training data (X_train, y_train), the batch size (12), and the number of epochs (200). We also pass in the validation data (X_val, y_val) so that we can track the model's performance on the validation set as it is trained.

The batch size is the number of samples we want to process before updating the model's weights and biases. The number of epochs is the number of times the model will go over the entire training dataset.

Once the model is trained, we can evaluate its performance on the test set.

```python
# Evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)
```

## Fin
And that is it! We have now successfully built and trained a deep learning model to classify the species of Iris from their measurements. The model achieved an accuracy of 97% on the test set, which is pretty good for a simple model.

I hope this tutorial gave you a better understanding of what goes into building and training a neural network and how to do it.

Try to see if you can make the model deeper, by adding more layers. You can also play with the hyperparameters to see how that changes the results. Maybe try graphing the loss and accuracy as the model trains. Another good thing to learn/use is early stopping, read about it [here](https://keras.io/api/callbacks/early_stopping/).


> Quesitons or want to chat about this post? Shoot me an [email](mailto://lewiswatson55@hotmail.co.uk) or message/tweet over on twitter: [![Twitter](https://img.shields.io/twitter/follow/LewisNWatson?style=flat)](https://twitter.com/LewisNWatson)

