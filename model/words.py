
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
import time
import pickle
from pytorch_i3d import InceptionI3d

# Global variables
i3d = None
nlp = None
params = None


wlasl_dict = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_once():
    global i3d, nlp, params, wlasl_dict

    weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    size = 100
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(size)
    i3d.load_state_dict(torch.load(weights, map_location=device))
    i3d = i3d.to(device)
    if torch.cuda.device_count() > 1:
        i3d = nn.DataParallel(i3d)
    i3d.eval()

    wlasl_dict = {}
    with open('preprocess/wlasl_class_list.txt') as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value

    return True

def run_on_tensor(ip_tensor):
    global i3d, wlasl_dict

    if i3d is None:
        raise RuntimeError("Model not loaded. Call load_models_once() first.")

    ip_tensor = ip_tensor[None, :].to(device)
    t = ip_tensor.shape[2]

    with torch.no_grad():
        per_frame_logits = i3d(ip_tensor)
        predictions = F.interpolate(per_frame_logits, size=t, mode='linear', align_corners=False)

    predictions = predictions.transpose(2, 1).cpu()
    frame_logits = predictions.detach().numpy()[0][0]
    all_scores = F.softmax(torch.from_numpy(frame_logits), dim=0).numpy()

    confidence = float(np.max(all_scores))
    predicted_index = int(np.argmax(all_scores))
    if confidence > 0.4:
        predicted_word = wlasl_dict.get(predicted_index, "")
        print(f"Actual Predicted Word: {predicted_word}, Confidence: {confidence:.4f}")
        return predicted_word
    else:
        return " "

class SignLanguageProcessor:
    def __init__(self):
        self.frames = []
        self.offset = 0
        self.text_list = []
        self.word_list = []
        self.sentence_tokens = []
        self.text_count = 0
        self.batch = 40
        self.last_prediction_time = time.time()

    def process_frame(self, frame_bytes):
        global nlp, params

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return "[frame decode failed]"
        



        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = (processed_frame / 255.) * 2 - 1

        self.frames.append(processed_frame)
        self.offset += 1

        if len(self.frames) > self.batch:
            self.frames.pop(0)

        if self.offset % 20 == 0 and len(self.frames) == self.batch:
            current_time = time.time()
            cooldown_period = 5.0
            if current_time - self.last_prediction_time > cooldown_period:
                input_tensor = torch.from_numpy(np.asarray(self.frames, dtype=np.float32).transpose([3, 0, 1, 2]))
                text = run_on_tensor(input_tensor)
                if(text != ' '):
                    print(f"{text}")
                    self.last_prediction_time = current_time
                    return text
