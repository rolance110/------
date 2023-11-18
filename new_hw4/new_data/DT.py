import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取資料集train.csv
df = pd.read_csv('new_hw4/new_data/train.csv')
# 用head()查看前幾筆資料
print(df.head())

# 查看資料的摘要統計
print(df.describe())

# 查看每個欄位的數據類型和缺失值
print(df.info())

# 視覺化特徵分佈
sns.set(style="whitegrid") # 設定圖表背景和網格線的風格

# # 以箱形圖查看數值特徵的分佈
# numeric_features = df.select_dtypes(include=[np.number]).columns
# for feature in numeric_features:
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(x=feature, data=df)
#     plt.title(f'Distribution of {feature}')
#     plt.show()

# 以直方圖查看連續特徵的分佈
numeric_features = df.select_dtypes(include=[np.number]).columns
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    print(feature)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# # 以計數圖查看離散特徵的分佈
# categorical_features = df.select_dtypes(include=[object]).columns
# for feature in categorical_features:
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x=feature, data=df)
#     plt.title(f'Count of {feature}')
#     plt.show()

# # 相關性矩陣
# correlation_matrix = df.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title('Correlation Matrix')

# plt.show()
