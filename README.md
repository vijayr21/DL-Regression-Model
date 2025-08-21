# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: VIJAY R

### Register Number: 212223240178
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X, y, color="magenta")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Genearted data for linear Regression')
plt.show()
```
# Initialize the Model, Loss Function, and Optimizer

```
class Model(nn.Module):
  def __init__(self,input_features,output_features):
    super().__init__()
    self.linear=nn.Linear(input_features,output_features)

  def forward(self,x):
    return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("\nName: VIJAY R ")
print("Register No: 212223240178")
print(f'Initial Weight: {initial_weight:.8f}')
print(f'Initial Bias: {initial_bias:.8f}')

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 100
losses = []

for epoch in range(1, epochs+1):
  optimizer.zero_grad()
  y_pred = model(X)
  loss = loss_function(y_pred, y)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  print(f'Epoch: {epoch:2} loss: {loss.item():10.8f}, '
        f'weight: {model.linear.weight.item():10.8f}, '
        f'bias: {model.linear.bias.item():10.8f}')
plt.plot(range(epochs), losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.show()

final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName: VIJAY R")
print("Registor No: 212223240178")
print(f"\nFinal Weight: {final_weight:.8f} , Final Bias: {final_bias:.8f}")

x1 = torch.tensor([X.min().item(), X.max().item()])
y1 = x1 * final_weight + final_bias

plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trained Model: Best-Fit Line")
plt.legend()
plt.show()

x_new = torch.tensor([120.0])
y_new_pred = model(x_new).item()
print("\nName: VIJAY R")
print("Registor No: 212223240178")
print(f"Predicted for X = 120: {y_new_pred:.8f}")


```

### Dataset Information
<img width="821" height="627" alt="image" src="https://github.com/user-attachments/assets/e3dc5c8e-9152-4b48-9f78-c20f30aa80eb" />

<img width="322" height="101" alt="image" src="https://github.com/user-attachments/assets/0c4ec23e-9086-4815-95d0-7dd04296e4d6" />


### OUTPUT
Training Loss Vs Iteration Plot

<img width="911" height="632" alt="image" src="https://github.com/user-attachments/assets/f5cf1ba2-558a-47f6-a533-323409194cd8" />

Best Fit line plot

<img width="821" height="625" alt="image" src="https://github.com/user-attachments/assets/6898c5ff-81d0-48c5-baff-4efd6cab988e" />

<img width="562" height="115" alt="image" src="https://github.com/user-attachments/assets/f05f2868-1fd6-451c-b3e1-eba3778ab79d" />


### New Sample Data Prediction

<img width="428" height="96" alt="image" src="https://github.com/user-attachments/assets/369e4441-acd4-4137-9c10-ab251d00af48" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
