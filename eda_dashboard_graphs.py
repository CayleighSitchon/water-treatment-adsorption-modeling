import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('synthetic_dataset_langmuir.csv')

# Create figure with 3 subplots (2 rows, 2 columns - last slot empty)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 1. Scatter plot: adsorbent mass vs removal %
sns.scatterplot(data=df, x='m', y='removal_pct', alpha=0.6, ax=axs[0, 0])
axs[0, 0].set_title('Adsorbent Mass vs Removal %')
axs[0, 0].set_xlabel('Adsorbent Mass (g)')
axs[0, 0].set_ylabel('Removal Percentage (%)')
axs[0, 0].grid(True)

# 2. Histogram: distribution of removal %
sns.histplot(df['removal_pct'], bins=30, kde=True, color='skyblue', ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Removal Percentage')
axs[0, 1].set_xlabel('Removal Percentage (%)')
axs[0, 1].set_ylabel('Frequency')

# 3. Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=axs[1, 0])
axs[1, 0].set_title('Correlation Heatmap')

# Remove the last empty subplot (bottom right)
fig.delaxes(axs[1,1])

plt.suptitle('Adsorption Data Exploratory Dashboard', fontsize=16, y=0.95)
plt.show()
