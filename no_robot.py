import pyrealsense2 as rs
import numpy as np
import cv2
import time
import mediapipe as mp
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

def gesture_speed(hand_sign):
    global robot_speed, speed_var_gesture, prev_hand_sign, count_gesture, count_frame, Stop, Go
    if prev_hand_sign == hand_sign:
        count_gesture += 1
    else:
        count_gesture = 0

    if count_gesture >= 7 or Stop or Go:
        count_frame += 1
        if hand_sign == "Open_Palm" or Stop:
            if robot_speed > 0.0 and count_frame >= 3:
                robot_speed = round(robot_speed - 0.1, 1)
                speed_var_gesture.set(f"{int(robot_speed * 100)}%")
                Stop = True
                count_frame = 0
            elif robot_speed <= 0.0:
                Stop = False
        elif hand_sign == "Pointing_Up" or Go:
            if robot_speed < 1.0 and count_frame >= 3:
                robot_speed = round(robot_speed + 0.1, 1)
                speed_var_gesture.set(f"{int(robot_speed * 100)}%")
                Go = True
                count_frame = 0
            elif robot_speed >= 1.0:
                Go = False
        elif hand_sign == "Thumbs_Up":
            if robot_speed < 1.0 and count_frame >= 10:
                robot_speed = round(robot_speed + 0.1, 1)
                speed_var_gesture.set(f"{int(robot_speed * 100)}%")
                count_frame = 0
        elif hand_sign == "Thumbs_Down":
            if robot_speed > 0.0 and count_frame >= 10:
                robot_speed = round(robot_speed - 0.1, 1)
                speed_var_gesture.set(f"{int(robot_speed * 100)}%")
                count_frame = 0
        else:
            pass    # proizvoljno
    else:
        count_frame = 0
        Stop = False
        Go = False
    prev_hand_sign = hand_sign

def starting_speed(speed):
    global robot_speed
    while robot_speed != speed:
        set_speed(speed, 0)
        # print(robot_speed)
        time.sleep(0.1)

def quit():
    global Quit
    starting_speed(0.0)
    try:
        pipeline.stop()
    except:
        pass
    try:
        cap.release()
    except:
        pass
    try:
        pose.close()
    except:
        pass
    try:
        hands.close()
    except:
        pass
    cv2.destroyAllWindows()
    root.destroy()
    Quit = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)
    args = parser.parse_args()
    return args

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_landmarks(image, hand_landmarks, mp_hands, mp_drawing):
    # Draw landmarks and connections using MediaPipe's utility
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,  # Pass the hand landmarks here
        mp_hands.HAND_CONNECTIONS,  # Draw connections between landmarks
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def set_speed(x, *args):
    global robot_speed, count, prev_x, speed_var, Nula
    # print(robot_speed)
    if Nula:
        if robot_speed == 0.0:
            Nula = False
        else:
            robot_speed = round(robot_speed - 0.1, 1)
            if not args:
                speed_var.set(f"{int(robot_speed * 100)}%")
        return
    if robot_speed == 0.0 and x < 0.8:  # vraćanje u zonu 2
        count = 0
        return
    if x == robot_speed:
        count = 0
        return
    if prev_x == x:
        count += 1
        if count >= 2:  # 2 kadra zaredom moraju dati istu brzinu da ju robot prihvati
            if x == 0.0:
                Nula = True
                robot_speed = round(robot_speed - 0.1, 1)
            elif x < robot_speed:
                robot_speed = round(robot_speed - 0.1, 1)
            else:
                robot_speed = round(robot_speed + 0.1, 1)
            if not args:
                speed_var.set(f"{int(robot_speed * 100)}%")
            count = 0
    else:
        count = 1
    prev_x = x

def camera_check():
    try:
        # Configure color stream only
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        return True

    except RuntimeError:
        return False

def update_frame():
    global prev_time, prev_y, pipeline_started, pose
    try:
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        fps = round(fps, 2)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            root.after(10, update_frame)
            return

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_image)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            right_foot_landmark = results.pose_landmarks.landmark[31]  # right heel 29
            left_foot_landmark = results.pose_landmarks.landmark[32]  # left heel 30

            # desna y-koordinata
            if right_foot_landmark.visibility < 0.5:
                right_y = None
            elif right_foot_landmark.y > 1 or right_foot_landmark.y < 0:
                right_y = None
            else:
                right_y = right_foot_landmark.y

            # lijeva y-koordinata
            if left_foot_landmark.visibility < 0.5:
                left_y = None
            elif left_foot_landmark.y > 1 or left_foot_landmark.y < 0:
                left_y = None
            else:
                left_y = left_foot_landmark.y

            # srednja y-koordinata
            try:
                y = (right_y + left_y) / 2
            except TypeError:
                y = None
            try:
                dy = y - prev_y
            except TypeError:
                dy = None
            prev_y = y

            # brzina točke na slici
            try:
                w = dy / (current_time - prev_time)
            except TypeError:
                w = None

            # stvarna udaljenost
            try:
                x = r0 / (2 * y - 1)
            except TypeError:
                x = None

            # stvarna brzina
            try:
                v = (-2 * r0 * w) / (2 * y - 1) ** 2
            except:
                v = None

            # algoritam
            if x is None:
                if robot_speed >= 0.6:  # covjek izlazi iz vidnog polja
                    set_speed(1.0)
                else:  # zona 0
                    set_speed(0.0)

            elif x >= r2:  # okolina
                set_speed(1.0)

            elif x < r2 and x >= r1:  # zona 2
                if v is not None:
                    if -v <= vm:
                        set_speed(0.8)
                    elif -v >= (vt + vbh) / 2:
                        set_speed(0.0)
                    elif -v >= (vbh + vh) / 2:
                        set_speed(0.3)
                    else:
                        set_speed(0.6)

            else:  # zona 1
                if v is not None:
                    if -v <= vm:
                        set_speed(0.3)
                    elif -v >= (vbh + vh) / 2:
                        set_speed(0.0)
                    else:
                        set_speed(0.1)
        else:  # nema covjeka
            if robot_speed != 0.0:
                set_speed(1.0)

        prev_time = current_time

        color_image = draw_info(color_image, fps, 0, -1)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (960, 720))
        img = Image.fromarray(color_image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        root.after(10, update_frame)
    except:
        return

def destroy_all_widgets(root):
    for widget in root.winfo_children():
        widget.destroy()

def safety():
    global robot_speed, count, prev_x, speed_var, Nula, printw, pipeline_started, pose
    global prev_time, prev_y, pipeline, mp_drawing, mp_pose, root, video_label

    starting_speed(0.0)

    try:
        cap.release()
    except:
        pass
    try:
        hands.close()
    except:
        pass
    cv2.destroyAllWindows()

    count = 0
    prev_x = 1.0
    Nula = False

    printw = False
    pipeline_started = False
    pose = None

    destroy_all_widgets(root)
    right_frame = tk.Frame(root, width=480, height=720, bg="#2C3E50")
    right_frame.grid(row=0, column=0, padx=(10, 820), pady=10)
    button_frame = tk.Frame(root, width=480, height=720, bg="#2C3E50")
    button_frame.grid(row=1, column=0, padx=702, pady=437)

    connect = tk.StringVar()
    connect.set("Waiting for the camera connection...")

    connect_font = font.Font(family="Helvetica", size=30, weight="bold")
    connect_label = tk.Label(right_frame, textvariable=connect, font=connect_font, bg="#2C3E50", fg="#ECF0F1")
    connect_label.pack()

    quit_button = tk.Button(button_frame, text="QUIT", font=("Helvetica", 20, "bold"),
                            command=quit,
                            bg="#E74C3C", fg="#ECF0F1", padx=20, pady=10)
    quit_button.pack(pady=100)

    try:
        while True:
            try:
                # Configure color stream only
                pipeline = rs.pipeline()
                config = rs.config()

                # Get device product line for setting a supporting resolution
                pipeline_wrapper = rs.pipeline_wrapper(pipeline)
                pipeline_profile = config.resolve(pipeline_wrapper)
                device = pipeline_profile.get_device()
                device_product_line = str(device.get_info(rs.camera_info.product_line))
                destroy_all_widgets(root)
                break

            except:
                # if not printw:
                #     print("Čekam spajanje s kamerom...")
                # printw = True
                root.update()
                if Quit:
                    raise(KeyboardInterrupt)

        found_rgb = False
        while True:
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    found_rgb = True
                    break
            if found_rgb:
                break
        config.enable_stream(rs.stream.color, res_h, res_v, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        pipeline_started = True

        starting_speed(1.0)

        prev_time = time.time()
        prev_y = None

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        left_frame = tk.Frame(root, width=960, height=720, bg="#2C3E50", padx=10, pady=10)
        left_frame.grid(row=0, column=0, padx=10, pady=10)
        right_frame = tk.Frame(root, width=480, height=720, bg="#34495E", padx=50, pady=10)
        right_frame.grid(row=0, column=1, padx=10, pady=(50, 10))

        video_label = tk.Label(left_frame, bg="#2C3E50")
        video_label.pack(pady=(40, 10))

        robospeed = tk.StringVar()
        robospeed.set("Robot speed:")
        robospeed_font = font.Font(family="Helvetica", size=40, weight="bold")
        robospeed_label = tk.Label(right_frame, textvariable=robospeed, font=robospeed_font, bg="#34495E", fg="#ECF0F1")
        robospeed_label.pack()

        speed_var = tk.StringVar()
        speed_var.set(f"{int(robot_speed * 100)}%")

        speed_font = font.Font(family="Helvetica", size=80, weight="bold")
        speed_label = tk.Label(right_frame, textvariable=speed_var, font=speed_font, bg="#34495E", fg="#ECF0F1")
        speed_label.pack(pady=20)

        switch_button = tk.Button(right_frame, text="GESTURE RECOGNITION", font=("Helvetica", 20, "bold"), command=gesture_recognition,
                                  bg="#3498DB", fg="#ECF0F1", padx=20, pady=10)
        switch_button.pack(pady=(30, 10))

        quit_button = tk.Button(right_frame, text="QUIT", font=("Helvetica", 20, "bold"),
                                  command=quit,
                                  bg="#E74C3C", fg="#ECF0F1", padx=20, pady=10)
        quit_button.pack(pady=(0, 10))

        root.after(10, update_frame)
        root.mainloop()

    except KeyboardInterrupt:
        print("Program interrupted")

    finally:
        try:
            pipeline.stop()
        except:
            pass
        try:
            pose.close()
        except:
            pass
        cv2.destroyAllWindows()

def gesture_recognition():
    global robot_speed, speed_var_gesture, count_gesture, prev_hand_sign, Stop, Go
    starting_speed(0.0)
    try:
        try:
            pipeline.stop()
        except:
            pass
        try:
            pose.close()
        except:
            pass
        cv2.destroyAllWindows()

        destroy_all_widgets(root)
        count_gesture = 0
        prev_hand_sign = None
        Stop = False
        Go = False

        right_frame = tk.Frame(root, width=480, height=720, bg="#2C3E50")
        right_frame.grid(row=0, column=0, padx=(10, 900), pady=10)
        button_frame = tk.Frame(root, width=480, height=720, bg="#2C3E50")
        button_frame.grid(row=1, column=0, padx=702, pady=437)

        disconnect = tk.StringVar()
        disconnect.set("Please disconnect the camera...")

        disconnect_font = font.Font(family="Helvetica", size=30, weight="bold")
        disconnect_label = tk.Label(right_frame, textvariable=disconnect, font=disconnect_font, bg="#2C3E50", fg="#ECF0F1")
        disconnect_label.pack()

        quit_button = tk.Button(button_frame, text="QUIT", font=("Helvetica", 20, "bold"),
                                command=quit,
                                bg="#E74C3C", fg="#ECF0F1", padx=20, pady=10)
        quit_button.pack(pady=100)

        while camera_check():
            root.update()
            if Quit:
                raise(KeyboardInterrupt)
        destroy_all_widgets(root)

        args = get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height
        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        use_brect = True

        # Camera preparation ###############################################################
        cap = cv2.VideoCapture(cap_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils  # Initialize drawing utils
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)

        # Create frame for video feed
        left_frame = tk.Frame(root, width=960, height=720, bg="#2C3E50", padx=10, pady=10)
        left_frame.grid(row=0, column=0, padx=30, pady=10)
        right_frame = tk.Frame(root, width=480, height=720, bg="#34495E", padx=100, pady=10)
        right_frame.grid(row=0, column=1, padx=180, pady=(0,50))

        # Create label for video feed
        video_label = tk.Label(left_frame, bg="#2C3E50")
        video_label.pack()

        with Image.open("gesturenotes.png") as img:
            resized_image = img.resize((480, 331))
            photo = ImageTk.PhotoImage(resized_image)

        image_label = tk.Label(left_frame, image=photo, bg="#2C3E50")
        image_label.pack(pady=10)

        robospeed = tk.StringVar()
        robospeed.set("Robot speed:")
        robospeed_font = font.Font(family="Helvetica", size=40, weight="bold")
        robospeed_label = tk.Label(right_frame, textvariable=robospeed, font=robospeed_font, bg="#34495E", fg="#ECF0F1")
        robospeed_label.pack()

        speed_var_gesture = tk.StringVar()
        speed_var_gesture.set(f"{int(robot_speed * 100)}%")

        speed_font = font.Font(family="Helvetica", size=80, weight="bold")
        speed_label = tk.Label(right_frame, textvariable=speed_var_gesture, font=speed_font, bg="#34495E", fg="#ECF0F1")
        speed_label.pack(pady=20)

        switch_button = tk.Button(right_frame, text="SAFETY", font=("Helvetica", 20, "bold"), command=safety,
                                  bg="#2ECC71", fg="#ECF0F1", padx=20, pady=10)
        switch_button.pack(pady=(30, 10))

        quit_button = tk.Button(right_frame, text="QUIT", font=("Helvetica", 20, "bold"),
                                command=quit,
                                bg="#E74C3C", fg="#ECF0F1", padx=20, pady=10)
        quit_button.pack(pady=(0, 10))

        def update_frame():
            ret, image = cap.read()
            if not ret:
                root.after(10, update_frame)
                return

            image = cv2.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            # Processing landmarks and gestures
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(-1, 0, pre_processed_landmark_list, pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == "Not applicable":  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, hand_landmarks, mp_hands, mp_drawing)
                    debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id],
                                                 point_history_classifier_labels[most_common_fg_id[0][0]])
                    gesture_speed(keypoint_classifier_labels[hand_sign_id])
            else:
                point_history.append([0, 0])
                gesture_speed(None)

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, cvFpsCalc.get(), 0, -1)

            # Convert image to display in Tkinter
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(debug_image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            try:
                video_label.configure(image=imgtk)
                root.after(10, update_frame)
            except:
                pass

        # Start updating frames
        update_frame()

        def on_closing():
            cap.release()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except KeyboardInterrupt:
        print("Program prekinut")

    finally:
        try:
            pipeline.stop()
        except:
            pass
        try:
            hands.close()
        except:
            pass
        cv2.destroyAllWindows()

# send speed 0!!!
def start():
    global root
    root = tk.Tk()
    root.title("RealSense Viewer")

    root.configure(bg="#2C3E50")
    root.attributes("-fullscreen", True)

    def toggle_fullscreen(event=None):
        root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    root.bind("<Escape>", toggle_fullscreen)

    button_frame = tk.Frame(root, bg="#2C3E50")
    button_frame.pack(expand=True, pady=(100,0))

    safety_button = tk.Button(button_frame, text="SAFETY", font=("Helvetica", 20, "bold"), command=safety, bg="#2ECC71",
                              fg="#ECF0F1", padx=20, pady=10)
    gesture_recognition_button = tk.Button(button_frame, text="GESTURE RECOGNITION", font=("Helvetica", 20, "bold"),
                                           command=gesture_recognition, bg="#3498DB", fg="#ECF0F1", padx=20, pady=10)
    quit_button = tk.Button(button_frame, text="QUIT", font=("Helvetica", 20, "bold"),
                            command=quit,
                            bg="#E74C3C", fg="#ECF0F1", padx=20, pady=10)

    safety_button.pack(pady=20)
    gesture_recognition_button.pack(pady=20)
    quit_button.pack(pady=(120,20))

    root.mainloop()

r0, r1, r2 = 2, 4, 6    # 2, 3.5, 5.5
vm, vh, vbh, vt = 0.1, 1.2, 1.7, 2.5
res_h, res_v = 640, 480

global robot_speed, Nula, Quit
robot_speed = 0.0
count = 0
prev_x = 1.0
Nula = False
Quit = False

if __name__ == "__main__":
    start()
