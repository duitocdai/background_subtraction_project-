import cv2
import streamlit as st
import numpy as np
import pygame
from datetime import datetime, time as time_lib
import telepot
import os
import time as time_module

# Khởi tạo pygame mixer để phát âm thanh
pygame.mixer.init(frequency=22050, size=-16, channels=2)
alert_sound = "C:\\internproject\\warning.mp3"  # Đường dẫn đến tệp âm thanh cảnh báo

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

    # Điều khiển nguồn video
    source_option = st.sidebar.selectbox("Chọn nguồn video", ("Webcam", "Video File"))
    video_file_path = None
    if source_option == "Video File":
        uploaded_file = st.sidebar.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_file_path = os.path.join("temp_video.mp4")
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    # Điều khiển webcam hoặc video
    start_button = st.sidebar.button("Bắt đầu")
    stop_button = st.sidebar.button("Dừng")
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
        current_time = datetime.now().time()
        return start_time <= current_time <= end_time

    def is_bbox_overlap(x, y, w, h, x1, y1, x2, y2):
        return not (x > x2 or x + w < x1 or y > y2 or y + h < y1)

    if start_button:
        if source_option == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_file_path) if video_file_path else None
        
        if cap is None or not cap.isOpened():
            st.error("Không thể mở nguồn video")
        else:
            st.sidebar.success("Nguồn video đã được bật!")
            
            sound_playing = False

            start_time_bg = time_module.time()
            while time_module.time() - start_time_bg < 10:
                ret, frame = cap.read()
                if not ret:
                    st.error("Không thể đọc frame từ video")
                    break
                fgMask = backSub.apply(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", caption="Ghi nhớ background...")
                time_module.sleep(0.1)

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Không thể đọc frame từ video")
                        break
                    
                    fgMask = backSub.apply(frame)
                    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    motion_detected = False

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_contour) > 500:
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            motion_detected = True

                            if is_bbox_overlap(x, y, w, h, x1, y1, x2, y2) and is_within_time_range(start_time, end_time):
                                if not sound_playing:
                                    pygame.mixer.music.load(alert_sound)
                                    pygame.mixer.music.play()
                                    sound_playing = True

                                try:
                                    bot.sendMessage(chat_id, "Phát hiện chuyển động trong hộp!")
                                    st.success("Thông báo đã được gửi!")
                                except Exception as e:
                                    st.error(f"Error sending message: {e}")
                                
                                screenshot_path = "motion_detected.jpg"
                                cv2.imwrite(screenshot_path, frame)
                                try:
                                    with open(screenshot_path, "rb") as f:
                                        bot.sendPhoto(chat_id, f)
                                    os.remove(screenshot_path)
                                    st.success("Ảnh đã được gửi!")
                                except Exception as e:
                                    st.error(f"Error sending photo: {e}")
                        else:
                            if sound_playing:
                                pygame.mixer.music.stop()
                                sound_playing = False
                    else:
                        if sound_playing:
                            pygame.mixer.music.stop()
                            sound_playing = False

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", caption="Video Feed")
                    fgmask_placeholder.image(fgMask, caption="Foreground Mask")

                    if stop_button:
                        st.sidebar.success("Nguồn video đã được dừng!")
                        break

            finally:
                cap.release()
                pygame.mixer.quit()

                # Xóa tệp video tạm sau khi hoàn tất
                if video_file_path and os.path.exists(video_file_path):
                    os.remove(video_file_path)
