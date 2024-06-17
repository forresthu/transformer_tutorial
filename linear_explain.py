# The nn.Linear class is used to apply a linear transformation to the incoming data,
# i.e., y = xA^T + b, where A is the weight and b is the bias.

import torch
from torch import nn

# Define a Linear layer with 5 input features and 3 output features
linear_layer = nn.Linear(in_features=5, out_features=3)

# Create a 2D input tensor with 4 samples, each having 5 features
input_tensor = torch.randn(4, 5)

# Apply the linear layer to the input tensor
output_tensor = linear_layer(input_tensor)

print("Input Tensor:")
print(input_tensor)

print("\nOutput Tensor:")
print(output_tensor)

# Print trainable parameters
for name, param in linear_layer.named_parameters():
    print(f"\nParameter name: {name}")
    print(f"Parameter value: {param.data}")
    print(f"Parameter requires_grad: {param.requires_grad}")