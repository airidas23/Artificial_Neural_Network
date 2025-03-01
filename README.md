# Artificial Neural Network Project

## Project Overview
This project, titled "KTU LAB4_2", involves building and training an artificial neural network (ANN) to predict loan statuses based on a given dataset. The project is implemented using Jupyter Notebook and primarily uses TensorFlow and Keras libraries.

## Dataset
The dataset used in this project is `credit_risk_dataset.csv`, which contains information about individuals' loan applications and statuses. The dataset includes the following columns:
- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_status`
- `loan_percent_income`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

## Data Preprocessing
The data is first cleaned by handling missing values and converting categorical variables into dummy/indicator variables. The dataset is then split into training and testing sets.

## Model Architecture
The ANN model consists of:
- An input layer with 10 neurons and ReLU activation
- Two hidden layers with 5 neurons each and ReLU activation
- An output layer with a single neuron and a linear activation function

```python
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
```

## Training the Model
The model is compiled using the mean squared error loss function and the Adam optimizer. It is trained for 20 epochs with a batch size of 64.

```python
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
```

## Evaluation
The model's performance is evaluated using the mean squared error (MSE) metric. Additionally, cross-validation is performed to estimate the model's accuracy.

## Analysis Using Project Pictures
### Activation Functions and Their Derivatives
![image](https://github.com/user-attachments/assets/07ccd70e-ddee-46d7-8a93-fd6819d24904)(#)

### XOR with Different Activation Functions
![image](https://github.com/user-attachments/assets/3dc0550c-c611-47c6-bbb9-06499b26d8ff)(#)

### Training History
![image](https://github.com/user-attachments/assets/0cc6e52c-98d8-4041-a87a-3e5cc091c394)

### Cross-Validation Results
![image](https://github.com/user-attachments/assets/87dd5854-92f9-4e57-8f37-00fe538f9d70)

## Improvements
To increase the model's accuracy, the following changes were made:
- Added additional hidden layers
- Adjusted the learning rate
- Changed the activation function of the output layer to `sigmoid`

```python
def create_improved_model():
    model = Sequential()
    model.add(Dense(15, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
```

## Results
The improved model achieved an average accuracy of 0.79, which is a 4.44% improvement over the initial model. However, this did not meet the target of a 5% improvement.

## Conclusion
This project demonstrates the process of building, training, and evaluating an artificial neural network for loan status prediction. Despite improvements, further optimization is required to achieve the desired accuracy.

Feel free to explore the [notebook](https://github.com/airidas23/Artificial_Neural_Network/blob/master/Lab4_2.ipynb) for more details.
```

You can add this content by navigating to the repository and creating a new file named `README.md`.
