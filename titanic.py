import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# memvisualisasi 
sns.set(style="whitegrid")
plt.style.use('ggplot')

# load dataset
df = sns.load_dataset('titanic')
print("Data Loaded")

# Eklorasi awal
print("\n--- Information Dataset ---")
print(df.info())

print("\n--- 5 Data Teratas ---")
print(df.head())

print("\n--- Ringkasan Statistik ---")
print(df.describe(include='all'))

# check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# membersihkan data
df.drop(['deck'], axis=1, inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.dropna(subset=['embark_town'], inplace=True)

# Analisis Univariat
# Visualisasi Distribusi fitur numerik
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Distribusi gender
sns.countplot(data=df, x='sex')
plt.title("Distribusi Penumpang Berdasarkan Jenis Kelamin")
plt.show()

# Analisis Bivariat
# Survival Rate by Sex
sns.countplot(data=df, x='sex', hue='survived')
plt.title("Tingkat Kelangsungan Hidup Berdasarkan Jenis Kelamin")
plt.show()

# Survival Rate by Class
sns.countplot(data=df, x='class', hue='survived')
plt.title("Tingkat Kelangsungan Hidup Berdasarkan Kelas")
plt.show()

# Age vs Survived
plt.figure(figsize=(10, 6))
sns.boxplot(x='survived', y='age', data=df)
plt.title("Distribusi Usia Berdasarkan Kelangsungan Hidup")
plt.show()

# Heatmap Korelasi numerik
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap Korelasi")
plt.show()