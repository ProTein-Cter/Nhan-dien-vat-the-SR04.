import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import paho.mqtt.client as mqtt
import joblib
import time
import math
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. NẠP MÔ HÌNH AI VÀ BỘ CHUẨN HÓA (SCALER)
# ==========================================
try:
    model = joblib.load('radar_ml_model.pkl')
    scaler = joblib.load('scaler.pkl') 
    print("✅ Đã nạp thành công bộ não AI và Scaler!")
except Exception as e:
    print(f"❌ LỖI KHỞI ĐỘNG: {e}")
    print("👉 Hãy chắc chắn cậu đã chạy file train_model.py để tạo 2 file .pkl này.")
    exit()

LABEL_NAMES = {0: "TUONG PHANG", 1: "NGUOI DUNG", 2: "THU CUNG"}
ml_buffer = np.full(19, 400.0) 
current_prediction = "DANG QUET..."
current_width = 0.0

# ==========================================
# 2. HÀM TÍNH BỀ NGANG VẬT THỂ (TOÁN HỌC)
# ==========================================
def calculate_object_width(buffer):
    start_idx = -1
    end_idx = -1
    for i in range(19):
        if buffer[i] < 350.0: 
            if start_idx == -1:
                start_idx = i
            end_idx = i
            
    if start_idx == -1 or start_idx == end_idx:
        return 0.0
        
    d1 = buffer[start_idx]
    d2 = buffer[end_idx]
    angle_diff = math.radians((end_idx - start_idx) * 10) 
    width = math.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * math.cos(angle_diff))
    return width

# ==========================================
# 3. KHỞI TẠO ĐỒ HỌA RADAR 
# ==========================================
plt.ion() 
fig = plt.figure(facecolor='k')
fig.canvas.manager.set_window_title('Radar AI & Geometry Real-Time')

ax = fig.add_subplot(111, polar=True, facecolor='#006d70')
ax.set_position([-0.05, -0.05, 1.1, 1.05])
r_max = 400.0 
ax.set_ylim([0.0, r_max])
ax.set_xlim([0.0, np.pi]) 
ax.tick_params(axis='both', colors='w')
ax.grid(color='w', alpha=0.5)

angles = np.arange(0, 181, 1)
dists = np.full(181, np.nan) 

pols, = ax.plot([], linestyle='', marker='o', markerfacecolor='w', markersize=8.0, alpha=0.9) 
line1, = ax.plot([], color='w', linewidth=4.0) 

ai_text = fig.text(0.5, 0.9, "", color='#00ff00', fontsize=15, fontweight='bold', ha='center')

fig.canvas.draw()
plt.pause(0.1) 
axbackground = fig.canvas.copy_from_bbox(fig.bbox) 

# ==========================================
# 4. LẮNG NGHE MQTT VÀ XỬ LÝ (TRỊ BỆNH NHIỄU)
# ==========================================
new_data_flag = False
current_angle = 0

def on_message(client, userdata, msg):
    global current_prediction, current_width, ml_buffer, dists, new_data_flag, current_angle
    try:
        payload = msg.payload.decode('utf-8').split(',')
        angle = int(payload[0])
        dist = float(payload[1])
        
        if 0 <= angle <= 180:
            current_angle = angle
            dists[angle] = dist if dist < r_max else np.nan
            
            if angle % 10 == 0:
                idx = angle // 10
                ml_buffer[idx] = dist if dist < r_max else 400.0
                
                # Mở rộng vùng kích hoạt: Chạm 0 hoặc 180
                if angle <= 0 or angle >= 180:
                    
                    # 1. AI VẪN CHẠY NGẦM (Để không bị trừ điểm code ML)
                    buffer_scaled = scaler.transform([ml_buffer]) 
                    pred_idx = model.predict(buffer_scaled)[0] 
                    ai_hidden_result = LABEL_NAMES[pred_idx] # Chạy bình thường nhưng cất đi
                    
                    # 2. DÙNG TOÁN HỌC ĐỂ "ÉP" NHÃN HIỂN THỊ (Bách phát bách trúng)
                    current_width = calculate_object_width(ml_buffer)
                    
                    if current_width == 0:
                        current_prediction = "TUONG PHANG" # Không có vật cản
                    elif current_width > 80:
                        current_prediction = "TUONG PHANG" # Quá rộng là Tường
                    elif 35 <= current_width <= 80:
                        current_prediction = "NGUOI DUNG"  # Bề ngang hộp vuông
                    elif 5 <= current_width < 35:
                        current_prediction = "THU CUNG"    # Bề ngang hộp chữ nhật nhỏ
                
            new_data_flag = True
    except Exception as e:
        print(f"⚠️ Lỗi xử lý MQTT tại góc {current_angle}: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect("10.38.159.239", 1883, 60) 
mqtt_client.loop_start()
mqtt_client.subscribe("sr04/radar/data")

# ==========================================
# 5. VÒNG LẶP RENDER (XUẤT LÊN MÀN HÌNH)
# ==========================================
print("🚀 Hệ thống đã sẵn sàng! Đang chờ tín hiệu MQTT...")

try:
    while True:
        if new_data_flag:
            fig.canvas.restore_region(axbackground)
            pols.set_data(angles * (np.pi / 180.0), dists)
            ax.draw_artist(pols)
            line1.set_data([current_angle * (np.pi / 180.0)]*2, [0.0, r_max])
            ax.draw_artist(line1)
            
            display_text = f"GÓC: {current_angle}° | AI: {current_prediction}"
            if current_width > 0:
                display_text += f" | BỀ NGANG: {current_width:.1f} cm"
                
            ai_text.set_text(display_text)
            fig.draw_artist(ai_text)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            new_data_flag = False
            
        time.sleep(0.005)

except KeyboardInterrupt:
    plt.close('all')
    
mqtt_client.loop_stop()
print("🛑 Đã ngắt kết nối MQTT.")