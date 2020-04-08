import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTM(nn.Module):

	def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
					num_layers=1):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers

		# Define the LSTM layer
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

		# Define the output layer
		self.linear = nn.Linear(self.hidden_dim, output_dim)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# This is what we'll initialise our hidden state as
		return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
				torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

	def forward(self, input):
		# Forward pass through LSTM layer
		# shape of lstm_out: [input_size, batch_size, hidden_dim]
		# shape of self.hidden: (a, b), where a and b both 
		# have shape (num_layers, batch_size, hidden_dim).

		'''
		print(type(input), '\nopopopopop\n\n')
		print(input.shape, len(input[0]), self.batch_size)
		#input = input.reshape(self.batch_size, -1, len(input[0][0]))

		print(input.shape, len(input[0]), self.batch_size)'''

		#input = torch.from_numpy(input)
		print(input.shape, 'lplplplplp')

		print(type(input[0][0][0]), '\n\nqqq\n', input.shape, self.hidden[0].shape)
		#lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size))
		lstm_out, self.hidden = self.lstm(input.float().view(len(input), self.batch_size, -1), self.hidden)
		
		# Only take the output from the final timetep
		# Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
		print('\npred\n')
		print(self.hidden, lstm_out.shape)
		#lstm_out = lstm_out.data.numpy()

		y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
		return y_pred.view(-1)



# model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)




# loss_fn = torch.nn.MSELoss(size_average=False)

# optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# #####################
# # Train model
# #####################

# hist = np.zeros(num_epochs)

# for t in range(num_epochs):
#     # Clear stored gradient
#     model.zero_grad()
    
#     # Initialise hidden state
#     # Don't do this if you want your LSTM to be stateful
#     model.hidden = model.init_hidden()
    
#     # Forward pass
#     y_pred = model(X_train)

#     loss = loss_fn(y_pred, y_train)
#     if t % 100 == 0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()

#     # Zero out gradient, else they will accumulate between epochs
#     optimiser.zero_grad()

#     # Backward pass
#     loss.backward()

#     # Update parameters
#     optimiser.step()
