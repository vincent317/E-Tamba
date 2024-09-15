import matplotlib.pyplot as plt

# Define the x-axis and y-axis data
x = [50, 100, 200, 400]
pythia = [1, 0.933, 1, 0.967]
mamba = [0.83, 0.46, 0.05, 0]
etamba = [0.7, 0.56, 0.38, 0.21]

# Create a new figure
plt.figure()

# Plot the lines
plt.plot(x, pythia, label='Pythia-1.4B')
plt.plot(x, mamba, label='Mamba-1.4B')
plt.plot(x, etamba, label='E-Tamba-1.1B')

# Add titles and labels
title = 'Copying Task'
plt.title(title)
plt.xlabel('Input sequence length')
plt.ylabel('Accuracy')

# Add a legend
plt.legend()

plt.savefig(title.replace(' ', '_'), dpi=300)