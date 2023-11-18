from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 假設你的資料集為df，其中包含所有特徵和目標變數
# X是特徵，y是目標變數

df = pd.read_csv('new_hw4/new_data/train.csv')

X = df.drop('fake', axis=1)
y = df['fake']

# 切割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立Decision Tree模型
model = DecisionTreeClassifier(random_state=42)

# 模型訓練
model.fit(X_train, y_train)

# 模型預測
y_pred = model.predict(X_test)

# 模型評估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# 定義Decision Tree模型
model = DecisionTreeClassifier(random_state=42)

# 定義超參數範圍
param_grid = {
    'max_depth': [None, 5, 10, 15],  # 嘗試不同的深度值
    'min_samples_split': [2, 5, 10],  # 嘗試不同的最小分割樣本數
    'min_samples_leaf': [1, 2, 4]  # 嘗試不同的最小葉子樣本數
}

# 使用GridSearchCV進行交叉驗證
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 找到最佳參數
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳參數重新訓練模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 使用測試集評估最佳模型性能
y_pred_best = best_model.predict(X_test)

# 模型評估
print("Accuracy (Best Model):", accuracy_score(y_test, y_pred_best))
print("Classification Report (Best Model):\n", classification_report(y_test, y_pred_best))

from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=['Real', 'Fake'])
plt.show()