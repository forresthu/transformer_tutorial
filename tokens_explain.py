import tiktoken
import torch

num_return_sequences = 5
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
print(tokens)
tokens = torch.tensor(tokens, dtype=torch.long)
print(tokens)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
print(tokens)