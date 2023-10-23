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
observation = 2

while True:
    if repeat:
        splits = int(sys.argv[1])
    else:
        splits = int(sys.argv[1])
    densityCube_p = torch.zeros(1, device=aqua_device)
    key1 = torch.tensor([0,1,2,3], device=aqua_device)
    key1 = torch.reshape(key1, [1, -1, 1])
    densityCube_p = densityCube_p + torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25], device=aqua_device).reshape(key1.shape))
    for i in range(1, 3000 + 1):
        drawnChar = torch.tensor([0,1,2,3], device=aqua_device)
        drawnChar = torch.reshape(drawnChar, [1, 1, -1])
        densityCube_p = densityCube_p + torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125], device=aqua_device).reshape(drawnChar.shape))
        densityCube_p = densityCube_p + torch.log(observation == drawnChar+key1)
        densityCube_p = torch.logsumexp(densityCube_p, 2, keepdim=True)

    expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
    z_expDensityCube = torch.sum(expDensityCube_p)
    posterior_key1 = expDensityCube_p.sum([0, 2]) / z_expDensityCube
    key1 = key1.flatten()
    posterior_key1 = posterior_key1.flatten()

    if repeat == False:
        break

    repeat = False
#print((key1.flatten() * posterior_key1).sum().item())
print(key1.flatten())
print(posterior_key1)

end.record()
print(start.elapsed_time(end) * MS_TO_S)
