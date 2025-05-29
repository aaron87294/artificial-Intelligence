import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("/workspaces/artificial-Intelligence/Week10/qb_stats.csv")


# (a) Find the mean of each numerical column
mean_values = df.select_dtypes(include='number').mean()
print("Mean of each numerical column:")
print(mean_values)

# (b) Find the standard deviation of each numerical column
std_values = df.select_dtypes(include='number').std()
print("\nStandard deviation of each numerical column:")
print(std_values)

# (c) Create a histogram of the number of yards
plt.figure(figsize=(8, 5))
plt.hist(df['yds'], bins=15, edgecolor='black')
plt.title('Histogram of Passing Yards')
plt.xlabel('Passing Yards')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_yards.png")
plt.show()

# (d) Create a boxplot of the number of touchdowns and identify outliers
plt.figure(figsize=(6, 5))
sns.boxplot(y=df['td'])
plt.title('Boxplot of Touchdowns')
plt.ylabel('Touchdowns')
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_touchdowns.png")
plt.show()

