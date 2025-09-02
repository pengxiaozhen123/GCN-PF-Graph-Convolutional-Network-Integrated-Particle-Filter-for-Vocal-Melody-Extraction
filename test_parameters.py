import mir_eval
import numpy as np
import torch
import os

np.set_printoptions(threshold=np.inf, precision=4)
torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None)
truth_dir = r"D:\1"
pre_dir = r'D:\2'
truth_files = os.listdir(truth_dir)
pre_files = os.listdir(pre_dir)
duc_length = []
overallpercent = []
raw_pitchpercent = []
raw_chromapercent = []
oaacc = []
pitchacc = []
chromaacc = []
for tf, pf in zip(truth_files, pre_files):
    print(tf, pf)
    print('*' * 20)
    ref_time, ref_fre = mir_eval.io.load_time_series(os.path.join(truth_dir, tf))
    duc_length.append(len(ref_time))
    est_time, est_fre = mir_eval.io.load_time_series(os.path.join(pre_dir, pf))
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(ref_time, ref_fre, est_time, est_fre)
    overall = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50)
    raw_pitch = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50)
    raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50)
    overallpercent.append(overall)
    raw_pitchpercent.append(raw_pitch)
    raw_chromapercent.append(raw_chroma)
    print("OA:", overall, "RPA:", raw_pitch, "RCA", raw_chroma)
for x, y, z, m in zip(duc_length, overallpercent, raw_pitchpercent, raw_chromapercent):
    oaacc.append(x * y)
    pitchacc.append(x * z)
    chromaacc.append(x * m)
print('*' * 20)
print("OA:{}\nRCA:{}\nRPA:{}".format(sum(oaacc) / sum(duc_length), sum(pitchacc) / sum(duc_length),
                                     sum(chromaacc) / sum(duc_length)))
totalmetrics = [sum(oaacc) / sum(duc_length), sum(pitchacc) / sum(duc_length), sum(chromaacc) / sum(duc_length)]
np.savetxt(r'D:\后处理文件夹\储存参数\OA.txt', np.array(overallpercent).T, fmt='%.3f')
np.savetxt(r'D:\后处理文件夹\储存参数\RPA.txt', np.array(raw_pitchpercent).T, fmt='%.3f')
np.savetxt(r'D:\后处理文件夹\储存参数\RCA.txt', np.array(raw_chromapercent).T, fmt='%.3f')
np.savetxt(r'D:\后处理文件夹\储存参数\total.txt', np.array(totalmetrics).T, fmt='%.3f')
