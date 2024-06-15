import json
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv() 

def translate(file_link):
    video_path = file_link
    video = extract_landmarks_from_video(video_path).astype(np.float32)
    video_preprocessed = predict_preprocess(video.copy())
    
    model = tf.keras.models.load_model(
        filepath=os.environ['MODEL_URL'], #TODO add model path here
        custom_objects={
            'EarlyLateDropout': EarlyLateDropout,
            'scce_with_ls': scce_with_ls,
            'add_dummy_channel': add_dummy_channel
        }
    )
    
    predictions = model.predict(np.expand_dims(video_preprocessed, axis=0))
    
    decoder_dict = load_decoder(os.environ['ENCODER_URL']) #TODO add encoder path here
    
    result = decode_top_prediction(predictions, decoder_dict)
    return result

def detect(detector, image, frame_timestamp=None):
    if frame_timestamp is None:
        return detector.detect(image)
    else:
        return detector.detect_for_video(image, frame_timestamp)

def process_landmarks(landmarks, filters, start_idx, landmarks_array):
    for i in filters:
        landmarks_array[start_idx] = [landmarks[i].x, landmarks[i].y]
        start_idx += 1
    return start_idx

def extract_landmarks_from_image(image, detectors, timestamp=None):
    hand_filters = list(range(21))
    pose_filters = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    face_filters = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    hand_filters_lend = len(hand_filters)
    pose_filters_len = len(pose_filters)
    face_filters_len = len(face_filters)
    total_filters_len = hand_filters_lend * 2 + pose_filters_len + face_filters_len
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    hands_detector, pose_detector, face_detector = detectors

    with ThreadPoolExecutor() as executor:
        detect_args = (mp_image, timestamp) if timestamp != None else (mp_image,)
        
        hands_future = executor.submit(detect, hands_detector, *detect_args)
        pose_future = executor.submit(detect, pose_detector, *detect_args)
        face_future = executor.submit(detect, face_detector, *detect_args)
        
        hand_result = hands_future.result()
        pose_result = pose_future.result()
        face_result = face_future.result()

    hand_landmarks = hand_result.hand_landmarks
    pose_landmarks = pose_result.pose_landmarks
    face_landmarks = face_result.face_landmarks

    landmarks_array = np.full((total_filters_len, 2), np.nan)
    arr_idx = 0
    
    if hand_landmarks:
        if hand_result.handedness[0].index == 1:
            arr_idx += hand_filters_lend
        
        for landmarks in hand_landmarks:
            arr_idx = process_landmarks(landmarks, hand_filters, arr_idx, landmarks_array)
        
        if arr_idx == hand_filters_lend:
            arr_idx += hand_filters_lend
    else:
        arr_idx += hand_filters_lend*2

    if pose_landmarks:
        arr_idx = process_landmarks(pose_landmarks[0], pose_filters, arr_idx, landmarks_array)
    else:
        arr_idx += pose_filters_len

    if face_landmarks:
        arr_idx = process_landmarks(face_landmarks[0], face_filters, arr_idx, landmarks_array)
    else:
        arr_idx += face_filters_len
    
    return landmarks_array

def extract_landmarks_from_video(video_path, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_duration_ms = 1000 / fps
    
    vision_running_mode = vision.RunningMode
    base_options = python.BaseOptions
    
    hands_options_video = vision.HandLandmarkerOptions(
        running_mode=vision_running_mode.VIDEO,
        min_hand_detection_confidence=0.5,
        base_options=base_options(model_asset_path=os.environ['HAND_URL']), #TODO add hand model path here
        num_hands=2
    )
    
    pose_options_video = vision.PoseLandmarkerOptions(
        running_mode=vision_running_mode.VIDEO,
        min_pose_detection_confidence=0.5,
        base_options=base_options(model_asset_path=os.environ['POSE_URL']) #TODO add pose model path here
    )
    
    face_options_video = vision.FaceLandmarkerOptions(
        running_mode=vision_running_mode.VIDEO,
        min_face_detection_confidence=0.5,
        base_options=base_options(model_asset_path=os.environ['FACE_URL']) #TODO add face model path here
    )
    
    hands_detector_video = vision.HandLandmarker.create_from_options(hands_options_video)
    pose_detector_video = vision.PoseLandmarker.create_from_options(pose_options_video)
    face_detector_video = vision.FaceLandmarker.create_from_options(face_options_video)
    VIDEO_DETECTORS = (hands_detector_video, pose_detector_video, face_detector_video)
    
    if start_frame < 1:
        start_frame = 1
    elif start_frame > total_frames:
        start_frame = 1
    
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    video_landmarks = np.zeros((end_frame - start_frame + 1, 95, 2), dtype=object)
    
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = int((frame_idx - 1) * frame_duration_ms)
        
        landmarks = extract_landmarks_from_image(frame, VIDEO_DETECTORS, timestamp)
        video_landmarks[frame_idx - start_frame] = landmarks
    
    cap.release()
    return video_landmarks

def padding(X, length=None, pad=0):
    if length is None:
        length = X.shape[0]
    
    if X.shape[0] > length:
        X_padded = X[:length]
    else:
        pad_length = length - X.shape[0]
        X_padded = np.pad(
            X, ((0, pad_length), (0, 0), (0, 0)),
            mode='constant', constant_values=pad
        )
            
    return X_padded

def remove_no_hands(video):
    frames_to_keep = []
    for i, frame in enumerate(video):
        hand_landmarks_data = frame[:43]
        if not np.all(np.isnan(hand_landmarks_data)):
            frames_to_keep.append(i)
    video_with_hands = video[frames_to_keep]
    return video_with_hands

def is_dominant_hand(video):

    left_hand_sum = np.sum(~np.isnan(video[:, slice(0, 21)]), axis=1)
    right_hand_sum = np.sum(~np.isnan(video[:, slice(21, 43)]), axis=1)

    left_dominant_count = np.sum(left_hand_sum >= right_hand_sum)
    right_dominant_count = np.sum(left_hand_sum < right_hand_sum)

    return left_dominant_count > right_dominant_count

def hflip(data):
    data[:, :, 0] = 1 - data[:, :, 0]
    return data

def predict_preprocess(video):
    if not is_dominant_hand(video):
        hflip(video)
    video = remove_no_hands(video)
    np.nan_to_num(video, copy=False, nan=0)
    video = padding(video, 64, -100)
    return video

class EarlyLateDropout(tf.keras.layers.Layer):
    def __init__(self, early_rate, late_rate, switch_epoch, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.early_rate = early_rate
        self.late_rate = late_rate
        self.switch_epoch = switch_epoch
        self.dropout = tf.keras.layers.Dropout(early_rate)
    
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = self.add_weight(name="train_counter", shape=[], dtype=tf.int64, aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        if training:
            dropout_rate = tf.cond(self._train_counter < self.switch_epoch, lambda: self.early_rate, lambda: self.late_rate)
            x = self.dropout(inputs, training=training)
            x = tf.keras.layers.Dropout(dropout_rate)(x, training=training)
            self._train_counter.assign_add(1)
        else:
            x = inputs
        return x

def add_dummy_channel(x, fill_value=0):
    dummy_channel_shape = tf.concat([tf.shape(x)[:-1], [1]], axis=0)
    dummy_channel = tf.fill(dummy_channel_shape, tf.cast(fill_value, x.dtype))
    result = tf.concat([x, dummy_channel], axis=-1)
    return result

def scce_with_ls(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, 201, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.5)

def load_decoder(json_file_path):
    with open(json_file_path, 'r') as json_file:
        label_to_int = json.load(json_file)
    
    int_to_label = {v: k for k, v in label_to_int.items()}
    return int_to_label

def decode_top_prediction(predictions, decoder_dict):
    top_prediction = np.argmax(predictions)
    top_label = decoder_dict[top_prediction]
    return top_label