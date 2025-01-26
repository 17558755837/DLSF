import json
from torch.utils.data import Dataset
from dataset.randaugment import RandomAugment
from torchvision import transforms
import torch
from PIL import Image
from torchvision.transforms.functional import hflip, resize
import re
from random import random as rand
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import pylab
from scipy.interpolate import interp1d


def sample_frames(video_path, frame_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, frame_size)
    cap.release()
    cv2.destroyAllWindows()
    return resized_frame


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.：，！'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


# def sample_mel(video_path, n_mels):
#     video_clip = VideoFileClip(video_path)
#     # 提取音频
#     audio_clip = video_clip.audio
#     # 保存音频文件
#     audio_path = video_path.replace('videos', 'Audio').replace('.mp4', '.wav')
#     audio_clip.write_audiofile(audio_path, codec='pcm_s16le')  # 保存为wav格式
#     y, sr = librosa.load(audio_path)
#     # 获取梅尔频谱
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
#     mel = librosa.power_to_db(S, ref=np.max)
#     # 定义滤波器边界（以梅尔频率为单位）
#     low_freq_mel = librosa.hz_to_mel(0)  # 最低频率
#     high_freq_mel = librosa.hz_to_mel(sr / 2)  # Nyquist频率
#     mel_points = np.linspace(low_freq_mel, high_freq_mel, num=n_mels)  # 梅尔频率点
#
#     low_pass = (mel_points < librosa.hz_to_mel(2000)).astype(float)
#     band_pass = ((mel_points >= librosa.hz_to_mel(2000)) & (mel_points <= librosa.hz_to_mel(5000))).astype(float)
#     high_pass = (mel_points > librosa.hz_to_mel(5000)).astype(float)
#
#     # 应用滤波器
#     low_channel = mel * low_pass[:, np.newaxis]
#     band_channel = mel * band_pass[:, np.newaxis]
#     high_channel = mel * high_pass[:, np.newaxis]
#     res = np.stack([low_channel, band_channel, high_channel], axis=0)
#
#     # 插值或裁剪梅尔频谱
#     def resize_mel(mel, target_length):
#         original_length = mel.shape[-1]
#         if original_length == target_length:
#             return mel
#         elif original_length < target_length:
#             # 插值增加时间维度的长度
#             x = np.linspace(0, 1, original_length)
#             new_x = np.linspace(0, 1, target_length)
#             resized_mel = np.zeros((mel.shape[0], mel.shape[1], target_length))
#             for i in range(mel.shape[0]):
#                 for j in range(mel.shape[1]):
#                     interp_func = interp1d(x, mel[i, j, :], kind='linear', fill_value="extrapolate")
#                     resized_mel[i, j, :] = interp_func(new_x)
#             return resized_mel
#         else:
#             # 裁剪时间维度
#             return mel[:, :, :target_length]
#
#     res = resize_mel(res, 2048)
#     # librosa.display.specshow(mel)
#     save_path = video_path.replace('videos', 'Audio').replace('.mp4', '.npy')
#     pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
#     pylab.close()
#     res = torch.FloatTensor(res)
#     return res


class DGSM(Dataset):
    def __init__(self, config, is_train=None):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        self.is_train = is_train
        if self.is_train:
            self.ann = []
            for f in config['train_file']:
                self.ann += json.load(open(f, 'r', encoding='utf-8'))
            self.max_words = config['max_words']
            self.is_train = is_train
            self.image_res = config['image_res']
            self.transform = train_transform
        else:
            self.ann = []
            for f in config['val_file']:
                self.ann += json.load(open(f, 'r', encoding='utf-8'))
            self.max_words = config['max_words']
            self.is_train = is_train
            self.image_res = config['image_res']
            self.transform = test_transform



    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index):
        res = self.ann[index]
        text = res['title']
        text_caption = pre_caption(text, self.max_words)
        fake_cls = res['fake_cls']
        fake_text_pos = res['fake_text_pos']
        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1

        img_dir = res['image_path']
        image = Image.open(img_dir).convert('RGB')
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(res['bbox'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                image = hflip(image)  # 水平反转
                do_hflip = True
            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)

        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                        center_y / self.image_res,
                        w / self.image_res,
                        h / self.image_res],
                        dtype=torch.float)


        video_path = res['video_path']
        reszied_frame = sample_frames(video_path)
        video = torch.FloatTensor(np.moveaxis(reszied_frame, 2, 0))
        audio_path = video_path.replace('videos', 'Audio').replace('mp4', 'npy')
        audio_res = np.load(audio_path)
        audio = torch.FloatTensor(audio_res)

        con_label_list = res['con_label']

        return text_caption, image, fake_text_pos_list, fake_cls,  video, audio,fake_image_box,W, H,con_label_list
