import numpy as np
import torch


def get_scores_reward(masks, scores, points_count, device='cpu', l1=1.0, l1_a=5.0, l2=1.0, l3=10, l3_a=0.5):
    reward = []
    areas = np.array([x["area"] for x in masks])
    q1=np.percentile(areas,25)
    for i,p in enumerate(zip(masks,scores,points_count)):
        p1 = torch.tensor(p[0]["predicted_iou"])
        p2 = torch.tensor(p[0]["stability_score"])
        p3 = (p1+p2)/2
        if p3 > 1.0:
            p3 = torch.tensor(0.9999)
        p3 = l1*torch.sigmoid(p3)**l1_a
        
        s1 = p[1].cpu()
        if s1 < 0 : s1 = 0.0
        s1 = l2*s1

        a1 = p[0]["area"]
        c1 = p[2]
        catf = a1 < q1
        ca = catf*l3_a*c1/a1 + (1-catf)*c1/a1
        ca = l3*torch.tensor(ca)

        l = p3+s1+ca
        
        reward.append(l.type(torch.float32).to(device))
    return reward

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]