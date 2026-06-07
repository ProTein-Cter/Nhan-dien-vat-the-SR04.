import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

LABEL_NAMES = {0: "TUONG PHANG", 1: "NGUOI DUNG", 2: "THU CUNG"}

print("🔄 BƯỚC 1: ĐANG NẠP DỮ LIỆU...")
try:
    df = pd.read_csv('radar_dataset.csv')
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    print(f"Tổng số mẫu dữ liệu: {len(df)}")

    print("\n✂️ BƯỚC 2: CHIA TẬP TRAIN / TEST (TỈ LỆ 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n🧠 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH DECISION TREE...")
    model = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    model.fit(X_train, y_train)

    print("\n📊 BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác tổng thể: {acc * 100:.2f}%\n")
    
    print("Đang vẽ Ma trận nhầm lẫn và lưu thành ảnh...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(LABEL_NAMES.values()), 
                yticklabels=list(LABEL_NAMES.values()))
    plt.title('MA TRAN NHAM LAN (CONFUSION MATRIX)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    
    print("\n💾 BƯỚC 5: XUẤT MÔ HÌNH...")
    joblib.dump(model, 'radar_ml_model.pkl')
    print("✅ Đã lưu bộ não AI: radar_ml_model.pkl sẵn sàng chạy thực tế!")

except Exception as e:
    print(f"❌ LỖI: Không tìm thấy file radar_dataset.csv. Chi tiết: {e}")