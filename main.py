import cv2
import streamlit as st
import numpy as np
import pygame
from datetime import datetime, time as time_lib
import telepot
import os
import time as time_module

# Initialize pygame mixer with a placeholder device
pygame.mixer.init()

# List of available audio devices (for simplicity, using generic device names)
audio_devices = ["Default Device", "Device 1", "Device 2"]

# Sidebar for audio device selection
selected_device = st.sidebar.selectbox("Chọn thiết bị âm thanh", audio_devices)

# Configure pygame mixer with the selected audio device
def initialize_audio_device(device):
    pygame.mixer.quit()  # Quit the current mixer before reinitializing
    if device == "Device 1":
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
    elif device == "Device 2":
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
    else:
        pygame.mixer.init()  # Default device

initialize_audio_device(selected_device)

alert_sound = "C:\\internproject\\warning.mp3"  # Path to alert sound file

# Streamlit User Interface
st.title("Phát hiện đối tượng với Trừ nền (Background Subtraction)")
st.sidebar.title("Cài đặt")

# Token and Chat ID fields for Telegram
token = "YOUR_TELEGRAM_BOT_TOKEN"  # Replace with your bot token
chat_id = st.sidebar.text_input("Nhập Chat ID")

if not chat_id:
    st.error("Vui lòng nhập Chat ID để chạy chương trình!")
else:
    bot = telepot.Bot(token)  # Initialize bot with token

    # Webcam control buttons
    start_button = st.sidebar.button("Bắt đầu Webcam")
    stop_button = st.sidebar.button("Dừng Webcam")
    start_time = st.sidebar.slider("Thời gian bắt đầu", value=time_lib(8, 0), format="HH:mm")
    end_time = st.sidebar.slider("Thời gian kết thúc", value=time_lib(18, 0), format="HH:mm")

    # ROI settings
    st.sidebar.subheader("Thiết lập hộp phát hiện")
    x1 = st.sidebar.slider("X1", 0, 640, 100)
    y1 = st.sidebar.slider("Y1", 0, 480, 100)
    x2 = st.sidebar.slider("X2", 0, 640, 400)
    y2 = st.sidebar.slider("Y2", 0, 480, 300)

    # Background Subtraction setup
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Placeholders for video and mask display
    frame_placeholder = st.empty()
    fgmask_placeholder = st.empty()

    # Time range check function
    def is_within_time_range(start_time, end_time):
        current_time = datetime.now().time()
        return start_time <= current_time <= end_time

    # Bounding box overlap check function
    def is_bbox_overlap(x, y, w, h, x1, y1, x2, y2):
        return not (x > x2 or x + w < x1 or y > y2 or y + h < y1)

    if start_button:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Không thể mở webcam")
        else:
            st.sidebar.success("Webcam đã được bật!")
            sound_playing = False
            notified = False

            # Capture initial background for 10 seconds
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
                    
                    fgMask = backSub.apply(frame)
                    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    motion_detected = False

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_contour) > 500:
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            motion_detected = True

                            if is_bbox_overlap(x, y, w, h, x1, y1, x2, y2):
                                cv2.putText(frame, "Inside Box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                if is_within_time_range(start_time, end_time):
                                    if not pygame.mixer.music.get_busy():
                                        pygame.mixer.music.load(alert_sound)
                                        pygame.mixer.music.play()
                                        sound_playing = True

                                        if not notified:
                                            try:
                                                bot.sendMessage(chat_id, "Phát hiện chuyển động trong hộp!")
                                                notified = True
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
                                            finally:
                                                time_module.sleep(0.5)
                                                if os.path.exists(screenshot_path):
                                                    os.remove(screenshot_path)
                            else:
                                notified = False
                        else:
                            notified = False
                        
                    else:
                        if sound_playing:
                            pygame.mixer.music.stop()
                            sound_playing = False

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")
                    fgmask_placeholder.image(fgMask, caption="Foreground Mask")

                    if stop_button:
                        st.sidebar.success("Webcam đã được dừng!")
                        break

            finally:
                cap.release()
                pygame.mixer.quit()
