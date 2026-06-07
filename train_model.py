import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. LOAD DATA
df = pd.read_csv('radar_dataset.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# 2. CHUẨN HÓA DỮ LIỆU (SCALING) - BẮT BUỘC TRONG ML
# Việc này giúp AI không bị "khớp" giữa các khoảng cách nhỏ (5cm) và lớn (400cm)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. CROSS-VALIDATION (ĐÁNH GIÁ ĐỘ ỔN ĐỊNH)
# Dùng CV để đảm bảo model không bị ăn may trên một tập test nhỏ
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"🎯 Độ ổn định (Cross-Validation Score): {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# 5. TRAIN & REPORT
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n📋 CHI TIẾT ĐỘ CHÍNH XÁC (CLASSIFICATION REPORT):")
print(classification_report(y_pred, y_test, target_names=["TUONG PHANG", "NGUOI DUNG", "THU CUNG"]))

# 6. LƯU MÔ HÌNH VÀ SCALER (RẤT QUAN TRỌNG)
# Phải lưu cả Scaler để khi chạy thực tế, ta chuẩn hóa dữ liệu đầu vào y hệt lúc train
joblib.dump(model, 'radar_ml_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Đã hoàn thiện bộ não AI & Scaler!")