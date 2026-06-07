import pandas as pd
import random

print("⏳ Đang tạo bộ dữ liệu mới (chống học vẹt)...")
columns = [f'Angle_{i}' for i in range(0, 181, 10)]
columns.append('Label')
dataset = []

# Nhãn 0: TƯỜNG PHẲNG (Khoảng cách đều, không có vật cản)
for _ in range(600):
    base_dist = random.uniform(20, 400) 
    row = [base_dist + random.uniform(-10, 10) for _ in range(19)]
    row = [min(x, 400.0) for x in row]
    row.append(0)
    dataset.append(row)

# Nhãn 1: NGƯỜI DÙNG (Vùng rộng 4-8 góc, khoảng cách ngẫu nhiên)
for _ in range(600):
    row = [400.0] * 19
    center_start = random.randint(2, 12)
    width = random.randint(4, 8)
    obj_dist = random.uniform(10, 300)
    for i in range(center_start, min(center_start + width, 19)):
        row[i] = obj_dist + random.uniform(-8, 8)
    row.append(1)
    dataset.append(row)

# Nhãn 2: THÚ CƯNG (Vùng hẹp 1-2 góc, khoảng cách ngẫu nhiên)
for _ in range(600):
    row = [400.0] * 19
    pet_pos = random.randint(1, 17)
    width = random.randint(1, 2)
    obj_dist = random.uniform(5, 200)
    for i in range(pet_pos, min(pet_pos + width, 19)):
        row[i] = obj_dist + random.uniform(-3, 3)
    row.append(2)
    dataset.append(row)

df = pd.DataFrame(dataset, columns=columns)
df = df.sample(frac=1).reset_index(drop=True) 
df.to_csv('radar_dataset.csv', index=False)
print("✅ Hoàn tất tạo 'radar_dataset.csv'.")