import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x should be of shape (batch_size, sequence_length, input_size)
        x = torch.reshape(x, [x.shape[0], x.shape[2], x.shape[1]])  # ???
        x = x.to(torch.float32)
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        return output

# Create a BiGRU model with input size 10, hidden size 20, 2 layers, and output size 1
# model = BiGRU(10, 20, 2, 1)

# # Convert the data to a form suitable for input into the BiGRU model
# data = ...  # Your data here
# data = data.unsqueeze(0)  # Add a batch dimension

# # Train the model
# optimizer = torch.optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = model(data)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
