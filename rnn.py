import torch
import torch.nn as nn
import torch.optim as optim

# Sample settings
input_size = 1      # Input feature dimension
hidden_size = 16    # RNN hidden units
output_size = 1     # Output dimension
sequence_length = 5 # Sequence length
num_layers = 1      # Number of RNN layers
num_epochs = 100    # Number of epochs for training
learning_rate = 0.01

# Toy dataset: Sequence of incremental values, target is next value
def generate_data(seq_len=sequence_length, num_samples=100):
    x = torch.linspace(0, 1, steps=seq_len + 1)
    data = []
    for _ in range(num_samples):
        start = torch.rand(1) * 0.5
        seq = start + 0.1 * torch.arange(seq_len + 1)
        data.append(seq.unsqueeze(-1))
    return torch.stack(data)

# Prepare data
dataset = generate_data()
inputs = dataset[:, :-1, :]  # All except last element
targets = dataset[:, 1:, :]  # Target is next value in sequence

# Define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

# Initialize model, loss, optimizer
model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing (Evaluation) on new unseen sequence
model.eval()
with torch.no_grad():
    test_seq = torch.linspace(0.5, 0.5 + 0.1*sequence_length, steps=sequence_length).unsqueeze(0).unsqueeze(-1)
    predicted = model(test_seq)
    print("Test input:", test_seq.squeeze().tolist())
    print("Predicted next values:", predicted.squeeze().tolist())
