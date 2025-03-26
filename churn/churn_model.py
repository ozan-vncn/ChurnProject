import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
df = pd.read_csv('Churn_Modelling.csv')

# Gereksiz sütunları kaldırma
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Kategorik değişkenleri dönüştürme
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Bağımlı ve bağımsız değişkenleri ayırma
X = df.drop('Exited', axis=1)
y = df['Exited']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri oluşturma
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Her model için eğitim ve değerlendirme
results = {}
for name, model in models.items():
    # Model eğitimi
    model.fit(X_train_scaled, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test_scaled)
    
    # Doğruluk hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Model Performans Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

# Modelleri doğruluklarına göre sıralama
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
print("\nModellerin Doğruluk Sıralaması:")
for model, accuracy in sorted_results.items():
    print(f"{model}: {accuracy:.4f}")

# Her model için özellik önem sıralaması
for name, model in models.items():
    print(f"\n{name} Özellik Önem Sıralaması:")
    
    if name == 'Random Forest':
        importance = model.feature_importances_
    elif name == 'Logistic Regression':
        # Logistic Regression için katsayıların mutlak değerlerini kullan
        importance = np.abs(model.coef_[0])
    else:  # XGBoost
        importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Özellik önem grafiği
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'{name} Özellik Önem Sıralaması')
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_feature_importance.png')
    plt.close() 