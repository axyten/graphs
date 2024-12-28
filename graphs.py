import matplotlib.pyplot as plt
import numpy as np

# Define the datasets
datasets = ["Alpaca Code Generation", "Anthropic HH dataset", "One Million Instructions"]

# Define the models
models = ["7b", "7b-chat", "13b-chat", "13b"]

# Define colors for each model
model_colors = {
    "7b": "tab:blue",
    "7b-chat": "tab:orange",
    "13b-chat": "tab:green",
    "13b": "tab:red",
}

# Generate some random data that resemble the plot
np.random.seed(42)  # for reproducibility

data = {
    "Alpaca Code Generation": {
        "7b": np.random.uniform(0.15, 0.7, 15),
        "7b-chat": np.random.uniform(0.05, 0.3, 15),
        "13b-chat": np.random.uniform(0.0, 0.4, 15),
        "13b": np.random.uniform(0.2, 0.95, 15),
    },
    "Anthropic HH dataset": {
        "7b": np.random.uniform(0.1, 0.6, 15),
        "7b-chat": np.random.uniform(0.1, 0.3, 15),
        "13b-chat": np.random.uniform(0.1, 0.3, 15),
        "13b": np.random.uniform(0.0, 0.95, 15),
    },
    "One Million Instructions": {
        "7b": np.random.uniform(0.05, 0.7, 15),
        "7b-chat": np.random.uniform(0.0, 0.35, 15),
        "13b-chat": np.random.uniform(0.0, 0.3, 15),
        "13b": np.random.uniform(0.0, 0.5, 15),
    },
}

# Create the plot
plt.figure(figsize=(8, 6))
for i, dataset in enumerate(datasets):
    for model in models:
      y_values= data[dataset][model]
      x_values= np.ones(len(y_values))*i
      x_values+=np.random.normal(0,0.03,len(y_values))
      plt.scatter(x_values, y_values, label=model, color=model_colors[model],s=20)


# Set labels and title
plt.ylabel("Token F1")
plt.xticks(range(len(datasets)), datasets)
plt.title("Jailbreak prompt performance by dataset")

# Add legend
plt.legend(loc='center right',bbox_to_anchor=(1.15,0.7))

# Adjust plot margins
plt.margins(x=0.1)

# Show the plot
plt.show()
