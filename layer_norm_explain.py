import torch
from torch.nn import LayerNorm

# Define a 2D input tensor with 5 samples, each having 3 features
input_tensor = torch.randn(5, 3)

# Create a LayerNorm module
layer_norm = LayerNorm(normalized_shape=input_tensor.shape[1])

# Apply LayerNorm to the input tensor
output_tensor = layer_norm(input_tensor)

# New code to print trainable parameters
# LayerNorm has two trainable parameters: weight and bias
# the size of weight and bias is equal to the number of features in the input tensor
# The default value of weight is 1 and bias is 0
for name, param in layer_norm.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Parameter value: {param.data}")
    print(f"Parameter requires_grad: {param.requires_grad}")
    print("\n")

print("Input Tensor:")
print(input_tensor)

print("\nOutput Tensor:")
print(output_tensor)