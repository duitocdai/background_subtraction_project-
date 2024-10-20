import cv2
import streamlit as st
import numpy as np
import pygame
from datetime import datetime, time as time_lib
import telepot
import os

# Khởi tạo âm thanh bằng pygame
pygame.mixer.init()
alert_sound = "C:\\internproject\\warning.mp3"  

# Thiết lập tiêu đề và giao diện
st.title("Phát hiện đối tượng với Trừ nền (Background Subtraction)")
st.sidebar.title("Cài đặt")

# Thêm các trường nhập token và chat_id
token = st.sidebar.text_input("Nhập Telegram Token", type="password")
chat_id = st.sidebar.text_input("Nhập Chat ID")

# Nếu không có token và chat_id, thông báo lỗi và không chạy chương trình
if not token or not chat_id:
    st.error("Vui lòng nhập Telegram Token và Chat ID để chạy chương trình!")
else:
    bot = telepot.Bot(token)

    # Tùy chọn để bật/tắt webcam
    start_button = st.sidebar.button("Bắt đầu Webcam")
    stop_button = st.sidebar.button("Dừng Webcam")
    start_time = st.sidebar.slider("Thời gian bắt đầu", value=time_lib(8, 0), format="HH:mm")
    end_time = st.sidebar.slider("Thời gian kết thúc", value=time_lib(18, 0), format="HH:mm")

    # Khởi tạo bộ trừ nền
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Đặt khung hiển thị video và mặt nạ nền trong Streamlit
    frame_placeholder = st.empty()
    fgmask_placeholder = st.empty()

    def is_within_time_range(start_time, end_time):
        """Kiểm tra xem thời gian hiện tại có nằm trong khoảng thời gian báo động hay không."""
        current_time = datetime.now().time()
        return start_time <= current_time <= end_time

    if start_button:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Không thể mở webcam")
        else:
            st.sidebar.success("Webcam đã được bật!")
            
            sound_playing = False  # Biến để theo dõi trạng thái âm thanh
            notified = False  # Biến để theo dõi trạng thái thông báo

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Không thể đọc frame từ webcam")
                    break
                
                # Áp dụng background subtraction
                fgMask = backSub.apply(frame)
                
                # Tìm các contour của các đối tượng di chuyển
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Biến để theo dõi xem có chuyển động hay không
                motion_detected = False

                # Vẽ bounding box cho các đối tượng
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Chỉ vẽ hộp cho các đối tượng lớn hơn ngưỡng này
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        motion_detected = True  # Phát hiện chuyển động

                # Nếu phát hiện chuyển động và đang trong thời gian đã chọn
                if motion_detected and is_within_time_range(start_time, end_time):
                    if not sound_playing:  # Nếu âm thanh chưa phát
                        pygame.mixer.music.load(alert_sound)  # Tải file âm thanh
                        pygame.mixer.music.play()  # Phát âm thanh
                        sound_playing = True  # Cập nhật trạng thái âm thanh
                        if not notified:  # Nếu chưa gửi thông báo
                            bot.sendMessage(chat_id, "Phát hiện chuyển động!")  # Gửi thông báo
                            notified = True  # Đánh dấu là đã gửi thông báo

                            # Lưu ảnh chụp màn hình và gửi lên Telegram
                            screenshot_path = "motion_detected.jpg"
                            cv2.imwrite(screenshot_path, frame)  # Lưu ảnh chụp màn hình
                            with open(screenshot_path, "rb") as f:
                                bot.sendPhoto(chat_id, f)  # Gửi ảnh lên Telegram
                            os.remove(screenshot_path)  # Xóa file ảnh sau khi gửi
                else:
                    if sound_playing:  # Nếu âm thanh đang phát nhưng không có chuyển động
                        pygame.mixer.music.stop()  # Dừng âm thanh
                        sound_playing = False  # Cập nhật trạng thái âm thanh
                        notified = False  # Đánh dấu lại để có thể gửi thông báo khi có chuyển động tiếp theo

                # Hiển thị video và mặt nạ nền trong Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi BGR sang RGB
                frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")
                fgmask_placeholder.image(fgMask, caption="Foreground Mask")

                # Dừng khi nhấn nút dừng webcam
                if stop_button:
                    cap.release()
                    st.sidebar.success("Webcam đã được dừng!")
                    break

        cap.release()
        cv2.destroyAllWindows()
