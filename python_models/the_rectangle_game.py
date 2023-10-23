import torch, time
import sys
import torch.distributions as tdist

MS_TO_S = 1/1000
aqua_device = torch.device("cuda:0")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ma_eps = 1.0E-9
start.record()
repeat = False
observedDataX = torch.tensor([0.4, 0.5, 0.46, 0.43], device=aqua_device)
observedDataX = torch.reshape(observedDataX, [-1, 1, 1, 1, 1])
observedDataY = torch.tensor([0.7, 0.4, 0.63, 0.51], device=aqua_device)
observedDataY = torch.reshape(observedDataY, [-1, 1, 1, 1, 1])

while True:
    if repeat:
        splits = int(sys.argv[1])
    else:
        splits = int(sys.argv[1])
    densityCube_p = torch.zeros(1, device=aqua_device)
    x1 = torch.arange(ma_eps, 1 + ma_eps, step=(1 - ma_eps)/splits, device=aqua_device)
    x1 = torch.reshape(x1, [1, -1, 1, 1, 1])
    x2 = torch.arange(ma_eps, 1 + ma_eps, step=(1 - ma_eps)/splits, device=aqua_device)
    x2 = torch.reshape(x2, [1, 1, -1, 1, 1])
    y1 = torch.arange(ma_eps, 1 + ma_eps, step=(1 - ma_eps)/splits, device=aqua_device)
    y1 = torch.reshape(y1, [1, 1, 1, -1, 1])
    y2 = torch.arange(ma_eps, 1 + ma_eps, step=(1 - ma_eps)/splits, device=aqua_device)
    y2 = torch.reshape(y2, [1, 1, 1, 1, -1])
    densityCube_p = densityCube_p + torch.log(1/ (1 - 0) * torch.logical_and(x1 >= 0, x1 <= 1))
    densityCube_p = densityCube_p + torch.log(1/ (1 - 0) * torch.logical_and(x2 >= 0, x2 <= 1))
    densityCube_p = densityCube_p + torch.log(1/ (1 - 0) * torch.logical_and(y1 >= 0, y1 <= 1))
    densityCube_p = densityCube_p + torch.log(1/ (1 - 0) * torch.logical_and(y2 >= 0, y2 <= 1))
    for i in range(1, 4 + 1):
        densityCube_p = densityCube_p + torch.log(torch.nan_to_num(1/ (x2 - x1), posinf=0, neginf=0) * torch.logical_and(observedDataX.view(-1)[i-1] >= x1, observedDataX.view(-1)[i-1] <= x2))
        densityCube_p = densityCube_p + torch.log(torch.nan_to_num(1/ (y2 - y1), posinf=0, neginf=0) * torch.logical_and(observedDataY.view(-1)[i-1] >= y1, observedDataY.view(-1)[i-1] <= y2))

    expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
    z_expDensityCube = torch.sum(expDensityCube_p)
    posterior_y1 = expDensityCube_p.sum([1, 2, 0, 4]) / z_expDensityCube
    posterior_x1 = expDensityCube_p.sum([0, 2, 3, 4]) / z_expDensityCube
    posterior_y2 = expDensityCube_p.sum([1, 2, 3, 0]) / z_expDensityCube
    posterior_x2 = expDensityCube_p.sum([1, 0, 3, 4]) / z_expDensityCube
    y1 = y1.flatten()
    posterior_y1 = posterior_y1.flatten()
    x1 = x1.flatten()
    posterior_x1 = posterior_x1.flatten()
    y2 = y2.flatten()
    posterior_y2 = posterior_y2.flatten()
    x2 = x2.flatten()
    posterior_x2 = posterior_x2.flatten()

    if repeat == False:
        break

    repeat = False
#print((y1.flatten() * posterior_y1).sum().item())
print((x1.flatten()))
print(posterior_x1)
#print((y2.flatten() * posterior_y2).sum().item())
#print((x2.flatten() * posterior_x2).sum().item())

end.record()
print(start.elapsed_time(end) * MS_TO_S)
