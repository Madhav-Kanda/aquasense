import time
starta = time.time()
import torch
import sys
import torch.distributions as tdist
print(time.time() - starta)

#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
#start.record()
#end.record()
MS_TO_S = 1/1000
ma_eps = 1.0E-9
aqua_device = torch.device("cpu")
posterior_b = [None] * 2
all_gt_b = [None] * 2
lowProb_b = [None] * 2
adaptiveLower_b = [None] * 2
adaptiveUpper_b = [None] * 2
posterior_o = [None] * 1
all_gt_o = [None] * 1
lowProb_o = [None] * 1
adaptiveLower_o = [None] * 1
adaptiveUpper_o = [None] * 1
repeat = False

while True:
    if repeat:
        splits = int(sys.argv[1])
    else:
        splits = int(sys.argv[1])
    densityCube_p = torch.zeros(1, device=aqua_device)
    b = [None] * 2
    b[0] = torch.tensor([0,1], device=aqua_device)
    b[0] = torch.reshape(b[0], [1, -1, 1, 1, 1])
    b[1] = torch.tensor([0,1], device=aqua_device)
    b[1] = torch.reshape(b[1], [1, 1, -1, 1, 1])
    o = [None] * 1
    o[0] = torch.arange(ma_eps, 200 + ma_eps, step=(200 - ma_eps)/splits, device=aqua_device)
    o[0] = torch.reshape(o[0], [1, 1, 1, -1, 1])
    x1 = torch.arange(50, 150 + ma_eps, step=(150 - 50)/splits, device=aqua_device)
    x1 = torch.reshape(x1, [1, 1, 1, 1, -1])
    densityCube_p = densityCube_p + torch.log((0.2) * (b[0]) + (1 - (0.2)) * (1 - (b[0])))
    densityCube_p = densityCube_p + torch.log((0.5) * (b[1]) + (1 - (0.5)) * (1 - (b[1])))
    densityCube_p_true = torch.tensor(0)
    densityCube_p_false = torch.tensor(0)
    densityCube_p_true = densityCube_p_true + torch.log(b[1] == 1)
    densityCube_p_false = densityCube_p_false + torch.log((0.2) * (b[1]) + (1 - (0.2)) * (1 - (b[1])))
    densityCube_p = densityCube_p + torch.log((b[0]).int() * torch.exp(densityCube_p_true) + (1 - (b[0]).int()) * torch.exp(densityCube_p_false))
    densityCube_p = densityCube_p + torch.log((1/ (200 - 0)) * torch.logical_and(o[0] >= 0, o[0] <= 200))
    densityCube_p = densityCube_p + torch.log((1/ (150 - 50)) * torch.logical_and(x1 >= 50, x1 <= 150))
    densityCube_p_true = torch.tensor(0)
    densityCube_p_false = torch.tensor(0)
    densityCube_p_true = densityCube_p_true + torch.log(torch.logical_and(o[0]>=x1-(50),o[0]<=x1)*2*(o[0]-(x1-(50)))/(50*(50+10)) + torch.logical_and(o[0]>x1,o[0]<=x1+10)*2*((x1+10)-(o[0]))/(10*(50+10)))
    densityCube_p_false = densityCube_p_false + torch.log(torch.logical_and(o[0]>=x1-(50),o[0]<=x1)*2*(o[0]-(x1-(50)))/(50*(50+50)) + torch.logical_and(o[0]>x1,o[0]<=x1+50)*2*((x1+50)-(o[0]))/(50*(50+50)))
    densityCube_p = densityCube_p + torch.log((b[0]).int() * torch.exp(densityCube_p_true) + (1 - (b[0]).int()) * torch.exp(densityCube_p_false))
    densityCube_p = densityCube_p + torch.log(b[0] == 1)

    expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
    z_expDensityCube = torch.sum(expDensityCube_p)
    posterior_b[1] = expDensityCube_p.sum([1, 0, 3, 4]) / z_expDensityCube
    posterior_x1 = expDensityCube_p.sum([1, 2, 3, 0]) / z_expDensityCube
    posterior_o[0] = expDensityCube_p.sum([1, 2, 0, 4]) / z_expDensityCube
    posterior_b[0] = expDensityCube_p.sum([0, 2, 3, 4]) / z_expDensityCube
    b[1] = b[1].flatten()
    posterior_b[1] = posterior_b[1].flatten()
    x1 = x1.flatten()
    posterior_x1 = posterior_x1.flatten()
    o[0] = o[0].flatten()
    posterior_o[0] = posterior_o[0].flatten()
    b[0] = b[0].flatten()
    posterior_b[0] = posterior_b[0].flatten()

    if repeat == False:
        break

    repeat = False
#torch.set_printoptions(precision=8, sci_mode=False)
#print((o[0].flatten()))
#print(posterior_o[0])

#print(start.elapsed_time(end) * MS_TO_S)
ends = time.time()
print(ends - starta)
