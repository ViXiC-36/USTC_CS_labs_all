import matplotlib.pyplot as plt

# Data for the lines
x_values = [1, 2, 3,]
y_values_red = [0.40, 0.93, 1.00]
y_values_blue = [0.20, 0.33, 0.60]

# Create the plot
plt.plot(x_values, y_values_red, color='red', marker='o', label='MLP')
plt.plot(x_values, y_values_blue, color='blue', marker='o', label='NN')

# Add labels and title
plt.xlabel('lg(epochs)')
plt.ylabel('prediction accuracy')
# y axis from 0 to 1
plt.ylim(0, 1.1)
plt.title('epochs on prediction accuracy\n(-lg(learning rate) = 3)')

# Add a legend
plt.legend()

# Display the plot
plt.show()
