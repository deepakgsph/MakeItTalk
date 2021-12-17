import datetime
import json
import os
import sox
import sys
import uuid
from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
from pydub import AudioSegment

sys.path.append("thirdparty/AdaptiveWingLoss")
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import time
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model
from base64 import b64encode
import boto3
import os
import subprocess
from timeit import default_timer as timer
import copy


gvs = {}

s3_bucket_name = "sph-brand-voice-models-aiuser"
input_bucket_folder = "api-data"
client = boto3.client('s3')
s3 = boto3.resource('s3')

app = Flask(__name__)

default_head_name = 'paint_boy.jpg'  # the image name (with no .jpg) to animate
ADD_NAIVE_EYE = True  # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False  # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 2.  # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 2.  # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.7  # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)


def add_silence(audio_in_file, audio_out_file):
    global gvs
    if "one_sec_segment" not in gvs:
       gvs["one_sec_segment"] = AudioSegment.silent(duration=1000)
    song = AudioSegment.from_wav(audio_in_file)

    # Add above two audio segments
    final_song = song + gvs["one_sec_segment"]

    # Either save modified audio
    final_song.export(audio_out_file, format="wav")


def random_filename(ext):
    basename = "sph_avatar"
    suffix = datetime.datetime.now().strftime("%y%m%d_") + uuid.uuid4().hex[:5]
    filename = "_".join([basename, suffix]) + str(ext)
    return filename


@app.route('/voice-to-video', methods=['GET', 'POST'])
def voice_to_video():
    start = timer()

    global gvs
    image_file = request.args.get('image_file')
    if image_file is None:
        image_file = default_head_name
    audio_file = request.args.get('audio_file')
    print(image_file, audio_file)
    if image_file not in gvs:
        gvs[image_file] = {}
    else:
        s3.meta.client.download_file(s3_bucket_name, input_bucket_folder + "/" + str(image_file),
                                     'examples/' + str(image_file))
    s3.meta.client.download_file(s3_bucket_name, input_bucket_folder + "/" + str(audio_file),
                                 'examples/' + str(audio_file))
    print(image_file, audio_file)

    if "parser" not in gvs[image_file]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--jpg', type=str, default='{}'.format(image_file))
        parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

        parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
        parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
        parser.add_argument('--load_a2l_C_name', type=str,
                            default='examples/ckpt/ckpt_content_branch.pth')  # ckpt_audio2landmark_c.pth')
        parser.add_argument('--load_G_name', type=str,
                            default='examples/ckpt/ckpt_116_i2i_comb.pth')  # ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

        parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
        parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
        parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
        parser.add_argument('--reuse_train_emb_list', type=str, nargs='+',
                            default=[])  # ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
        parser.add_argument('--add_audio_in', default=False, action='store_true')
        parser.add_argument('--comb_fan_awing', default=False, action='store_true')
        parser.add_argument('--output_folder', type=str, default='examples')

        parser.add_argument('--test_end2end', default=True, action='store_true')
        parser.add_argument('--dump_dir', type=str, default='', help='')
        parser.add_argument('--pos_dim', default=7, type=int)
        parser.add_argument('--use_prior_net', default=True, action='store_true')
        parser.add_argument('--transformer_d_model', default=32, type=int)
        parser.add_argument('--transformer_N', default=2, type=int)
        parser.add_argument('--transformer_heads', default=2, type=int)
        parser.add_argument('--spk_emb_enc_size', default=16, type=int)
        parser.add_argument('--init_content_encoder', type=str, default='')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
        parser.add_argument('--write', default=False, action='store_true')
        parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--emb_coef', default=3.0, type=float)
        parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
        parser.add_argument('--use_11spk_only', default=False, action='store_true')
        parser.add_argument('-f')

        gvs[image_file]["parser"] = parser

    if "opt_parser" not in gvs[image_file]:
        opt_parser = gvs[image_file]["parser"].parse_args()
        gvs[image_file]["opt_parser"] = opt_parser
    #if "img" not in gvs[image_file]:
    img = cv2.imread('examples/' + gvs[image_file]["opt_parser"].jpg)
    gvs[image_file]["img"] = img
    #if "predictor" not in gvs:
    predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
    gvs["predictor"] = predictor
    if "shapes" not in gvs[image_file]:
        shapes = gvs["predictor"].get_landmarks(gvs[image_file]["img"])
        gvs[image_file]["shapes"] = shapes

    if (not gvs[image_file]["shapes"] or len(gvs[image_file]["shapes"]) != 1):
        print('Cannot detect face landmarks. Exit.')
        exit(-1)
    shape_3d = gvs[image_file]["shapes"][0]

    if (gvs[image_file]["opt_parser"].close_input_face_mouth):
        util.close_input_face_mouth(shape_3d)

    shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 1.05 + np.mean(shape_3d[48:, 0])  # wider lips
    shape_3d[49:54, 1] += 0.  # thinner upper lip
    shape_3d[55:60, 1] -= 1.  # thinner lower lip
    shape_3d[[37, 38, 43, 44], 1] -= 2.  # larger eyes
    shape_3d[[40, 41, 46, 47], 1] += 2.  # larger eyes

    shape_3d, scale, shift = util.norm_input_face(shape_3d)

    au_data = []
    au_emb = []
    ains = glob.glob1('examples', '*.wav')
    #print(ains)
    ains = [item for item in ains if (item is not 'tmp.wav' and item == audio_file)]
    #ains.sort()
    print(ains)
    for i in range(len(ains)):
        add_silence("examples/" + ains[i], "examples/" + ains[i])
        # ains[i] = ains[i][:-4] + "_sil.wav"

    for ain in ains:
        os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
        shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

        # au embedding
        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        me, ae = get_spk_emb('examples/{}'.format(ain))
        au_emb.append(me.reshape(-1))

        print('Processing audio file', ain)
        c = AutoVC_mel_Convertor('examples')

        au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
                                                         autovc_model_path=gvs[image_file]["opt_parser"].load_AUTOVC_name)
        au_data += au_data_i

        print('Processed audio file', ain)
    if (os.path.isfile('examples/tmp.wav')):
        os.remove('examples/tmp.wav')

    # landmark fake placeholder
    fl_data = []
    rot_tran, rot_quat, anchor_t_shape = [], [], []
    for au, info in au_data:
        au_length = au.shape[0]
        fl = np.zeros(shape=(au_length, 68 * 3))
        fl_data.append((fl, info))
        rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
        rot_quat.append(np.zeros(shape=(au_length, 4)))
        anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

    with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
        pickle.dump(fl_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
        pickle.dump(au_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
        gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
        pickle.dump(gaze, fp)

    """
    if "model" not in gvs[image_file]:
        model = Audio2landmark_model(gvs[image_file]["opt_parser"], jpg_shape=shape_3d)
        gvs[image_file]["model"] = model
    """

    #model = copy.deepcopy(gvs[image_file]["model"])
    model = Audio2landmark_model(gvs[image_file]["opt_parser"], jpg_shape=shape_3d)

    if (len(gvs[image_file]["opt_parser"].reuse_train_emb_list) == 0):
        model.test(au_emb=au_emb)
    else:
        model.test(au_emb=None)

    fls = glob.glob1('examples', 'pred_fls_{}_audio_embed.txt'.format(audio_file[:-4], ))
    fls.sort()

    print(fls)

    for i in range(0, len(fls)):
        fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68, 3))
        fl[:, :, 0:2] = -fl[:, :, 0:2]
        fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

        if (ADD_NAIVE_EYE):
            fl = util.add_naive_eye(fl)

        # additional smooth
        fl = fl.reshape((-1, 204))
        fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
        fl[:, 48 * 3:] = savgol_filter(fl[:, 48 * 3:], 5, 3, axis=0)
        fl = fl.reshape((-1, 68, 3))

        ''' STEP 6: Imag2image translation '''
        """
        if "model_img" not in gvs[image_file]:
            model_img = Image_translation_block(gvs[image_file]["opt_parser"], single_test=True)
            gvs[image_file]["model_img"] = model_img
        """

        model_img = Image_translation_block(gvs[image_file]["opt_parser"], single_test=True)

        with torch.no_grad():
            # gvs[image_file]["model_img"].single_test(jpg=gvs[image_file]["img"], fls=fl, filename=fls[i], prefix=gvs[image_file]["opt_parser"].jpg.split('.')[0])
            model_img.single_test(jpg=gvs[image_file]["img"], fls=fl, filename=fls[i], prefix=gvs[image_file]["opt_parser"].jpg.split('.')[0])
            print('finish image2image gen')
        os.remove(os.path.join('examples', fls[i]))
        print("{} / {}: Landmark->Face...".format(i + 1, len(fls)), file=sys.stderr)
    print("Done!", file=sys.stderr)
    response = {}
    for ain in ains:
        OUTPUT_MP4_NAME = '{}_pred_fls_{}_audio_embed.mp4'.format(
            gvs[image_file]["opt_parser"].jpg.split('.')[0],
            ain.split('.')[0]
        )

        OUTPUT_MP4_NAME_CR = random_filename(".mp4")

        input_dir = "examples/"
        output_dir = "examples/"
        crop = True
        """
        for filee in os.listdir(input_dir):
            if filee.endswith(OUTPUT_MP4_NAME):
                print(filee)
        """
        input_name = input_dir + OUTPUT_MP4_NAME
        output_name = output_dir + OUTPUT_MP4_NAME_CR
        if crop:
            bashCommand = "sudo ffmpeg -i " + input_name + " -vf crop=256:256:256:0 -strict -2 -y " + output_name
        else:
            bashCommand = "sudo cp " + input_name + " " + output_name
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        s3.meta.client.upload_file('examples/{}'.format(OUTPUT_MP4_NAME_CR),
                                   s3_bucket_name, 'avatar/' + OUTPUT_MP4_NAME_CR)
        response["avatar_output"] = OUTPUT_MP4_NAME_CR
    response = jsonify(response)

    end = timer()
    print("Voice To Video Response Time ", end - start)

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1900, debug=True)