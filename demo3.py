import cv2
import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import librosa
from models import AT_net, AT_single
from models import VG_net 
import scipy.misc
import utils
import subprocess

import mark_paint

from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes
from yandex_speech import TTS
import imageio

from pydub import AudioSegment
import re
import requests
from bleach import clean
import html


import dlib


def audio_shift(wavepath,wavepath_new):

    sound = AudioSegment.from_file(wavepath, format="wav")
    
    octaves = -1.01
    
    new_sample_rate = int(sound.frame_rate * (1.1 ** octaves))
    
    lowpitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    
    newSound = lowpitch_sound.speedup(playback_speed=1.1, chunk_size=300, crossfade=95)
    
    newSound = newSound.set_frame_rate(48000)
    
    newSound.export(wavepath_new, format="wav")
    return True


def search_lang(text):
        search_lang = len(re.findall(r'[а-яА-Я]+'
        ,text
        ,flags=re.I|re.X|re.S))
        return search_lang

def load_data(url):
    
    headers = {
           'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; rv:12.0) Gecko/20100101 Firefox/12.0',
           'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Language' : 'en-en,ru;q=0.8,en-us;q=0.5,en;q=0.3',
           'Accept-Encoding' : 'gzip, deflate',
           'Connection' : 'keep-close',
          }
    
    try:
        txt = requests.get(url, timeout=4, allow_redirects=False,headers=headers)
    except requests.exceptions.RequestException as e:
        print("ERROR: " + str(e))
        return ""
        
    
    sl = search_lang(txt.text)
    
    if sl > 10:
        return txt.text
    
    try:
        t = txt.text.encode("ISO-8859-1").decode("utf-8")
        sl = search_lang(t)
    except:
        pass
    
    if sl > 10:
        return t    

    if txt.encoding == "ISO-8859-1":
        txt.encoding = 'windows-1251'

    try:
        if txt.encoding != "windows-1251":
            t = txt.text.encode(txt.encoding).decode("utf-8")
            sl = search_lang(t)
            if sl > 10:
                return t
        else:
            return txt.text
    except:
        return ""

    return txt.text

def clean_content(data):

    clean_text = clean(data, tags=[], strip=True, strip_comments=True)
    clean_text = html.unescape(clean_text).replace("\xa0"," ").replace(".",". ").replace(",",". ")
    
    clean_text = re.sub(r'\(.{,100}?\)'
             ,r' '
             ,clean_text
             ,flags=re.I|re.X|re.S)
    
    clean_text = re.sub(r'\s{2,}'
             ,r' '
             ,clean_text
             ,flags=re.I|re.X|re.S)
    
    return clean_text

def get_info(url):
    txt = load_data(url)
    title = re.findall(r'<title>\s*(.*?)\s*<\/title>'
             ,txt
             ,flags=re.I|re.X|re.S)[0]
    
    h1 = re.findall(r'<h1\s*class="b-material-head__title">\s*(.*?)\s*<\/h1>'
             ,txt
             ,flags=re.I|re.X|re.S)[0]
    
    news_content = re.findall(r'<div\s*class="b-material-wrapper\s*b-material-wrapper_art"[^>]*>\s*(.*?)\s*<div\s*class="b-material-wrapper__rubric">'
             ,txt
             ,flags=re.I|re.X|re.S)
    
    
    
    
    if type(news_content) == list:
        news_content = ' '.join(news_content)

    news_image = re.findall(r'<img\s*class="b-read-more__img"[^>]*src="(.*?)"[^>]*srcset=[^>]*>[^>]*</picture>'
             ,news_content
             ,flags=re.I|re.X|re.S)   
    
    if type(news_image) == list:
        news_image = news_image[0]
    
    news_content = clean_content(news_content)

    title = html.unescape(title).strip()
    h1 = html.unescape(h1).strip()
    news_content = html.unescape(news_content).strip()
    
    return {"title":title,"h1":h1,"news_content":news_content,"news_image":news_image}






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--lstm",
                     default=True)
    parser.add_argument("--vg_model",
                     type=str,
                     default="models/gen.pth")
    parser.add_argument("--at_model",
                     type=str,
                     default="models/dis.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="/results")
    parser.add_argument('-i','--in_file', type=str, default='test.wav')
    parser.add_argument('-d','--data_path', type=str, default='basics')
    parser.add_argument('-p','--person', type=str, default='test1.jpg')
    parser.add_argument('--device_ids', type=str, default='2')
    parser.add_argument('--num_thread', type=int, default=1)   
    parser.add_argument('-nt','--name_tts', type=str, default='anton_samokhvalov')
    parser.add_argument('-lt','--lang_tts', type=str, default='ru-RU')    
    parser.add_argument('-tt','--text_tts', type=str, default='')
    parser.add_argument('-sft','--shift', type=int, default=1)
    parser.add_argument('-nurl','--news_url', type=str, default='')    
    return parser.parse_args()
config = parse_args()

print("Config ==============================")
print(config)
print("=====================================")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('basics/shape_predictor_68_face_landmarks.dat')
ms_img = np.load('basics/mean_shape_img.npy')
ms_norm = np.load('basics/mean_shape_norm.npy')
S = np.load('basics/S.npy')

MSK = np.reshape(ms_norm, [1, 68*2])
SK = np.reshape(S, [1, S.shape[0], 68*2])

def multi2single(model_path, id):
    checkpoint = torch.load(model_path,map_location='cpu')
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def normLmarks(lmarks):
    norm_list = []
    idx = -1
    max_openness = 0.2
    mouthParams = np.zeros((1, 100))
    mouthParams[:, 1] = -0.06
    tmp = deepcopy(MSK)
    tmp[:, 48*2:] += np.dot(mouthParams, SK)[0, :, 48*2:]
    open_mouth_params = np.reshape(np.dot(S, tmp[0, :] - MSK[0, :]), (1, 100))
    if len(lmarks.shape) == 2:
        lmarks = lmarks.reshape(1,68,2)
    for i in range(lmarks.shape[0]):
        mtx1, mtx2, disparity = procrustes(ms_img, lmarks[i, :, :])
        mtx1 = np.reshape(mtx1, [1, 136])
        mtx2 = np.reshape(mtx2, [1, 136])
        norm_list.append(mtx2[0, :])
    pred_seq = []
    init_params = np.reshape(np.dot(S, norm_list[idx] - mtx1[0, :]), (1, 100))
    for i in range(lmarks.shape[0]):
        params = np.reshape(np.dot(S, norm_list[i] - mtx1[0, :]), (1, 100)) \
        - init_params - open_mouth_params
        
        predicted = np.dot(params, SK)[0, :, :] + MSK
        pred_seq.append(predicted[0, :])
    return np.array(pred_seq), np.array(norm_list), 1
   
def crop_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        (x, y, w, h) = utils.rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)

        shape = ((shape - np.array([new_x,new_y])) * scale)

        return roi, shape 
def generator_demo_example_lips(img_path):
    name = img_path.split('/')[-1]
    landmark_path = os.path.join('image/', name.replace('jpg', 'npy')) 
    region_path = os.path.join('image/', name.replace('.jpg', '_region.jpg')) 
    roi, landmark= crop_image(img_path)
    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

        template = np.load( 'basics/base_68.npy')
    else:
        template = np.load( 'basics/base_68_close.npy')
    pts2 = np.float32(template[27:45,:])
    pts1 = np.float32(landmark[27:45,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))

    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129,1:129,:]
    cv2.imwrite(region_path, dst)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        shape, _ ,_ = normLmarks(shape)
        np.save(landmark_path, shape)
        lmark= shape.reshape(68,2)
        name = region_path.replace('region.jpg','lmark.png')

        utils.plot_flmarks(lmark, name, (-0.2, 0.2), (-0.2, 0.2), 'x', 'y', figsize=(10, 10))
    return dst, lmark
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids

    result_dir = 'temp/' + config.in_file
    motion_dir = result_dir + '/motion/'
    
    os.mkdir(result_dir)
    os.mkdir(motion_dir)
    
    pca = torch.FloatTensor( np.load('basics/pca.npy')[:,:6])
    mean =torch.FloatTensor( np.load('basics/mean.npy'))
    decoder = VG_net()
    encoder = AT_net()
    
    state_dict2 = multi2single(config.vg_model, 1)

    decoder.load_state_dict(state_dict2)

    state_dict = multi2single(config.at_model, 1)
    encoder.load_state_dict(state_dict)

    encoder.eval()
    decoder.eval()
    test_file = result_dir + "/" + config.in_file + ".wav"
    test_file_old = result_dir + "/old_" + config.in_file + ".wav"
    if config.text_tts == "" and config.news_url != "":
        parse_news_content = get_info(config.news_url)['news_content']
    else:    
        parse_news_content = config.text_tts
    tts = TTS(config.name_tts, "wav", "000000-0000-0000-0000-00000000",config.lang_tts, emotion="neutral", speed=1)
    # test content
    tts.generate(parse_news_content[:1999])
    if config.shift == 1:
        tts.save(test_file_old) 
        audio_shift(test_file_old,test_file)
    else:
        tts.save(test_file)        

    example_image, example_landmark = generator_demo_example_lips( config.person)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])        
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    example_image = transform(example_image)

    example_landmark =  example_landmark.reshape((1,example_landmark.shape[0]* example_landmark.shape[1]))

    if config.cuda == True:
        example_image = Variable(example_image.view(1,3,128,128)).cuda()
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)) ).cuda()
    else:
        example_image = Variable(example_image.view(1,3,128,128))
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)))
    example_landmark = example_landmark * 5.0
    example_landmark  = example_landmark - mean.expand_as(example_landmark)
    example_landmark = torch.mm(example_landmark,  pca)
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    sound, _ = librosa.load(test_file, sr=44100)

    print ('=======================================')
    print ('Generate images')
    t =time.time()
    ind = 3
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        fake_lmark = encoder(example_landmark, input_mfcc)
        fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 6)
        example_landmark  = torch.mm( example_landmark, pca.t() ) 
        example_landmark = example_landmark + mean.expand_as(example_landmark)
        fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])) 
        fake_lmark = torch.mm( fake_lmark, pca.t() )
        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
    

        fake_lmark = fake_lmark.unsqueeze(0) 

  
        fake_lmark = fake_lmark.data.cpu().numpy()
        
        
        file_mark = result_dir + "/" + config.in_file + ".npy"
        file_mp4 = result_dir + "/" + config.in_file# + ".mp4"      
        np.save(file_mark, fake_lmark)
        mark_paint.mark_video(fake_lmark,motion_dir)



        cmd = 'ffmpeg -framerate 25 -i ' + motion_dir + '%d.png  -filter:v scale=512:-1 -c:v libx264 -pix_fmt yuv420p ' + file_mp4 +'.mp4'
        subprocess.call(cmd, shell=True)
        print('video done')
        
        cmd = 'ffmpeg -i '+ file_mp4 +'.mp4 -i '+ test_file +' -c:v copy -c:a aac -strict experimental '+ file_mp4 + '_result.mp4'        
        subprocess.call(cmd, shell=True) 
        print('video+audio done')

        return file_mark
    return False

print(test())

