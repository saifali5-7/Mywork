import pandas as pd
import matplotlib
# TclError se bachne ke liye image backend use karein
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Step 3: Prepare and Clean the Dataset ---
# File load karein
df = pd.read_excel("Balochistan Population Dataset.xlsx")

# Data Cleaning: Filter karna zaroori hai taake counts aur % mix na houn
# Hum sirf 'Working Age Population' use karenge jo actual population distribution dikhata hai
filtered_df = df[df['Indicator'] == 'Working Age Population'].copy()

print("--- Step 3: Data Cleaning ---")
print(f"Total rows after filtering: {len(filtered_df)}")
print("Missing values check:\n", filtered_df.isnull().sum())

# --- Step 4: Exploratory Data Analysis (EDA) ---
# 1. Summary Table (Descriptive Statistics)
print("\n--- Step 4: Summary Statistics (Mean, Median, Std Dev) ---")
print(filtered_df[['Total', 'Male', 'Female']].describe())

# 2. Bar Chart (Division-wise Trends)
plt.figure(figsize=(12, 8))
sns.barplot(data=filtered_df, x='Division', y='Total', estimator=sum, errorbar=None, palette='viridis')
plt.title('Total Working Age Population by Division - Balochistan')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('population_bar_chart.png')
print("\n[SUCCESS] Bar chart saved as 'population_bar_chart.png'")

# 3. Boxplot (Outlier Detection - Required in Assignment)
plt.figure(figsize=(8, 6))
sns.boxplot(y=filtered_df['Total'], color='skyblue')
plt.title('Population Distribution & Outlier Detection')
plt.savefig('population_boxplot.png')
print("[SUCCESS] Boxplot saved as 'population_boxplot.png'")

# --- Step 5: Data Modelling (Regression) ---
# Male aur Female numbers se Total predict karna
X = filtered_df[['Male', 'Female']]
y = filtered_df['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation (R-squared score)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print(f"\n--- Step 5: Regression Model Results ---")
print(f"Model R-squared Score (Accuracy): {accuracy:.3f}")
