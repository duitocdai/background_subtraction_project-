import cv2
import streamlit as st
import numpy as np
import pygame
from datetime import datetime, time as time_lib
import telepot
import os
import time as time_module

# Khởi tạo mixer pygame để phát âm thanh
pygame.mixer.init(frequency=22050, size=-16, channels=2)
alert_sound = "warning.mp3"  # Đường dẫn đến tệp âm thanh cảnh báo

# Giao diện người dùng Streamlit
st.title("Phát hiện đối tượng với Trừ nền (Background Subtraction)")
st.sidebar.title("Cài đặt")

# Các trường nhập cho token và chat ID của Telegram
token = "7791888401:AAH3HWAPvJO4blS_ls-LH_5sr5S8Uz68_78"  # Thay thế bằng token của bạn
chat_id = st.sidebar.text_input("Nhập Chat ID")  # Nhập chat ID

# Kiểm tra token và chat ID
if not chat_id:
    st.error("Vui lòng nhập Chat ID để chạy chương trình!")
else:
    bot = telepot.Bot(token)  # Khởi tạo bot với token

    # Điều khiển webcam
    start_button = st.sidebar.button("Bắt đầu Webcam")
    stop_button = st.sidebar.button("Dừng Webcam")
    start_time = st.sidebar.slider("Thời gian bắt đầu", value=time_lib(8, 0), format="HH:mm")
    end_time = st.sidebar.slider("Thời gian kết thúc", value=time_lib(18, 0), format="HH:mm")

    # Cài đặt ROI
    st.sidebar.subheader("Thiết lập hộp phát hiện")
    x1 = st.sidebar.slider("X1", 0, 640, 100)
    y1 = st.sidebar.slider("Y1", 0, 480, 100)
    x2 = st.sidebar.slider("X2", 0, 640, 400)
    y2 = st.sidebar.slider("Y2", 0, 480, 300)

    # Tạo bộ trừ nền
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Các placeholder để hiển thị video và mặt nạ
    frame_placeholder = st.empty()
    fgmask_placeholder = st.empty()

    def is_within_time_range(start_time, end_time):
        """Kiểm tra xem thời gian hiện tại có trong khoảng thời gian không."""
        current_time = datetime.now().time()
        return start_time <= current_time <= end_time

    def is_bbox_overlap(x, y, w, h, x1, y1, x2, y2):
        """Trả về True nếu các hộp bao quanh chồng lấp lên nhau."""
        return not (x > x2 or x + w < x1 or y > y2 or y + h < y1)

    if start_button:
        cap = cv2.VideoCapture(0)  # Mở webcam

        if not cap.isOpened():
            st.error("Không thể mở webcam")
        else:
            st.sidebar.success("Webcam đã được bật!")
            
            sound_playing = False  # Theo dõi trạng thái âm thanh
            notified = False  # Theo dõi trạng thái thông báo

            # Ghi lại background trong 10 giây đầu tiên
            start_time_bg = time_module.time()
            while time_module.time() - start_time_bg < 10:
                ret, frame = cap.read()
                if not ret:
                    st.error("Không thể đọc frame từ webcam")
                    break
                fgMask = backSub.apply(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", caption="Ghi nhớ background...")
                time_module.sleep(0.1)

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Không thể đọc frame từ webcam")
                        break
                    
                    # Áp dụng trừ nền
                    fgMask = backSub.apply(frame)
                    
                    # Tìm contours
                    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    motion_detected = False

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_contour) > 500:  # Kiểm tra diện tích contour
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hộp bao quanh
                            motion_detected = True

                            # Kiểm tra chồng lấp với hộp thông báo
                            if is_bbox_overlap(x, y, w, h, x1, y1, x2, y2):
                                cv2.putText(frame, "Inside Box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                if is_within_time_range(start_time, end_time):
                                    if not pygame.mixer.music.get_busy():  # Kiểm tra nếu âm thanh không đang phát
                                        pygame.mixer.music.load(alert_sound)
                                        pygame.mixer.music.play()
                                        sound_playing = True

                                        if not notified:
                                            try:
                                                bot.sendMessage(chat_id, "Phát hiện chuyển động trong hộp!")  # Gửi tin nhắn đến Telegram
                                                notified = True  # Đánh dấu đã thông báo
                                                st.success("Thông báo đã được gửi!")  # Phản hồi thông báo
                                            except Exception as e:
                                                st.error(f"Error sending message: {e}")  # Hiển thị lỗi nếu có
                                            
                                            # Lưu và gửi ảnh chụp
                                            screenshot_path = "motion_detected.jpg"
                                            cv2.imwrite(screenshot_path, frame)  # Lưu ảnh chụp
                                            try:
                                                with open(screenshot_path, "rb") as f:
                                                    bot.sendPhoto(chat_id, f)  # Gửi ảnh chụp đến Telegram
                                                os.remove(screenshot_path )
                                                st.success("Ảnh đã được gửi!")  # Phản hồi gửi ảnh
                                            except Exception as e:
                                                st.error(f"Error sending photo: {e}")  # Hiển thị lỗi gửi ảnh nếu có
                                            finally:
                                                time_module.sleep(0.5)  # Đợi một chút trước khi xóa
                                                if os.path.exists(screenshot_path):
                                                    os.remove(screenshot_path)  # Xóa tệp ảnh
                            else:
                                notified = False
                        else:
                            notified = False
                        
                    else:
                        if sound_playing:
                            pygame.mixer.music.stop()  # Dừng âm thanh nếu không có chuyển động
                            sound_playing = False

                    # Vẽ hộp phát hiện
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Hiển thị video và mặt nạ
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")
                    fgmask_placeholder.image(fgMask, caption="Foreground Mask")

                    # Dừng khi nhấn nút dừng
                    if stop_button:
                        st.sidebar.success("Webcam đã được dừng!")
                        break

            finally:
                cap.release()  # Giải phóng webcam
                pygame.mixer.quit()  # Dừng mixer pygame
