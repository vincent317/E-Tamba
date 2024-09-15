import matplotlib.pyplot as plt

# Define the x-axis and y-axis data
x = [20, 40, 80, 100]
pythia = [1, 0.96, 0.82, 0.7]
mamba = [0.72, 0.26, 0.09, 0.03]
etamba = [0.61, 0.49, 0.44, 0.33]

# Create a new figure
plt.figure()

# Plot the lines
plt.plot(x, pythia, label='Pythia-1.4B')
plt.plot(x, mamba, label='Mamba-1.4B')
plt.plot(x, etamba, label='E-Tamba-1.1B')

# Add titles and labels
title = 'Phonebook Retrieval Task'
plt.title(title)
plt.xlabel('Input sequence length')
plt.ylabel('Accuracy')

# Add a legend
plt.legend()

plt.savefig(title.replace(' ', '_'), dpi=300)