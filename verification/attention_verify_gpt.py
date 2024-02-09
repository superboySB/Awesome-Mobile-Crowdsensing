import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        # Linear transformations for query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Split the tensors into multiple heads
        query = query.view(query.size(0), -1, self.num_heads, self.head_dim)
        key = key.view(key.size(0), -1, self.num_heads, self.head_dim)
        value = value.view(value.size(0), -1, self.num_heads, self.head_dim)

        # Calculate scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to the values
        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.view(attended_values.size(0), -1, self.num_heads * self.head_dim)

        # Final linear transformation
        output = self.output_linear(attended_values)
        return output, attention_weights


# Generate synthetic data
batch_size = 32
input_dim = 64
num_heads = 4

agent_positions = torch.randn(batch_size, input_dim)
emergency_positions = torch.randn(batch_size, input_dim)

# Calculate L2 distance as labels
labels = torch.sqrt(torch.sum((agent_positions - emergency_positions) ** 2, dim=1))

# Initialize model and optimizer
model = MultiHeadAttention(input_dim, num_heads)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    predictions, weights = model(agent_positions, agent_positions, agent_positions)

    # Calculate loss
    loss = criterion(weights, labels.unsqueeze(1))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Use the trained model to get attention weights
attention_weights = model(agent_positions, agent_positions, agent_positions)
