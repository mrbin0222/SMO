import numpy as np
import torch


def get_scores_loss(masks, scores, device='cpu', a_f=0.25, g_f=2.0, l_f=0.01,  l2_f=15, c_f=2.0, e_f=0.007):
    loss = []
    a = np.array([x["area"] for x in masks])
    for i,p in enumerate(zip(masks,scores)):
        pt = torch.tensor([p[0]["predicted_iou"]])
        b1 = (1.0-pt)**g_f
        c1 = l_f*torch.log(torch.sigmoid(p[1]))**2
        tf = p[0]["area"] < a.mean()
        d1 = np.abs(tf*l2_f*(p[0]["area"]-a.mean())/a.std()**c_f)
        los = -a_f*b1*torch.log(pt)+c1.cpu()+d1
        if los < 0:
            los = torch.tensor([0.0])
        loss.append(los.to(device))
    return loss

def get_indices_of_values_above_threshold_2(values, threshold):
    return [i for i, v in enumerate(values) if v < threshold]