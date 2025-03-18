import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Flag to choose between t-SNE and PCA
use_tsne = True

# Load data from the JSON file
with open("aria_embeddings.json", "r") as f:
    data = json.load(f)

# Define the set of top composers
top_composers = {
    "chopin",
    "bach",
    "handel",
    "haydn",
    "tchaikovsky",
    "scriabin",
    "beethoven",
    "liszt",
    "mozart",
    "debussy",
    "schumann",
    "schubert",
    "satie",
    "rachmaninoff",
    "brahms",
    "ravel",
}


# Filter the data to include only entries for the top composers
filtered_data = [entry for entry in data if entry["composer"] in top_composers]

# Extract embeddings and composers from the filtered data
embeddings = np.array([entry["emb"] for entry in filtered_data])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
composers = [entry["composer"].capitalize() for entry in filtered_data]

# Perform dimensionality reduction based on the flag
if use_tsne:
    reducer = TSNE(
        n_components=2, perplexity=50, max_iter=2500, random_state=43
    )
    title = "t-SNE Visualization of Composer Embeddings"
    filename = "/home/loubb/work/aria/tsne_plot.png"
else:
    reducer = PCA(n_components=2)
    filename = "/home/loubb/work/aria/pca_plot.png"

embeddings_2d = reducer.fit_transform(embeddings)

# Create a DataFrame for plotting
df = pd.DataFrame(
    {
        "Dimension 1": embeddings_2d[:, 0],
        "Dimension 2": embeddings_2d[:, 1],
        "Composer": composers,
    }
)

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid", font="Helvetica")

# Create the scatter plot
plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(
    data=df,
    x="Dimension 1",
    y="Dimension 2",
    hue="Composer",
    palette="tab20",
    s=50,  # Marker size
    edgecolor="w",
    linewidth=0.5,
)

plt.xlabel(None)  # Remove x-axis label
plt.ylabel(None)  # Remove y-axis label

plt.xticks([])  # Remove numerical x-axis ticks
plt.yticks([])  # Remove numerical y-axis ticks

plt.grid(True, linestyle="--", linewidth=0.5)  # Keep the grid visible

# Ensure grid is properly aligned
plt.gca().set_aspect("auto")  # Prevent distortion
plt.gca().set_frame_on(True)  # Keep figure frame
# plt.gca().set_xticks(
#     np.linspace(df["Dimension 1"].min(), df["Dimension 1"].max(), num=6)
# )
# plt.gca().set_yticks(
#     np.linspace(df["Dimension 2"].min(), df["Dimension 2"].max(), num=6)
# )

# Move the legend outside the plot
plt.legend(
    bbox_to_anchor=(0, -0.38, 1, 0),
    loc="lower center",
    ncol=4,  # Arrange in multiple columns
    fontsize=20,  # Increase font size
    columnspacing=1.05,  # Increase the space between columns
    title_fontsize=20,
    title="Composer",
)

# Set plot title and labels
# plt.title(title)

# Save the plot as a high-resolution PNG file
plt.savefig(filename, dpi=300, bbox_inches="tight")

# Display the plot
plt.show()
