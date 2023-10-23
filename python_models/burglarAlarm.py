import torch, time
import sys
import distr

MS_TO_S = 1/1000
splits = int(sys.argv[1])
device = torch.device("cuda:0") # 
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ma_eps = 0.00001
start.record()

######################
p = torch.zeros(1, device=device)
earthquake, p_earthquake = flip(0.0001, device, [-1, 1, 1])
p = p + p_earthquake
burglary, p_burglary = flip(0.001, device, [1, -1, 1])
p = p + p_burglary
alarm = torch.logical_or(a, b)

phoneWorking1, p_phoneWorking1 = flip(0.7, device, [1, 1, -1], False)
p1 = torch.exp(p) * 
phoneWorking2, p_phoneWorking2 = flip(0.7, device, [1, 1, -1], False)
phoneWorking = phoneWorking1
true_prob = torch.prod(earthquake, p_earthquake,)
p = p + torch.log2(torch.prod( (earthquake) , p_phoneWorking1)

a = torch.reshape(a, [1, -1])
b = torch.arange(-10, 10 + ma_eps, step=20/splits,  device=device)
b = torch.reshape(b, [-1, 1])
apb = a + b
data = torch.tensor([-2.57251482,  0.33806206,  2.71757796,  1.09861336,  2.85603752,
        -0.91651351,  0.15555127, -2.68160347,  2.47043789,  3.47459025,
        1.63949862, -1.32148757,  2.64187513,  0.30357848, -4.09546231,
        -1.50709863, -0.99517866, -2.0648892 , -2.40317949,  3.46383544,
        0.91173696,  1.18222221,  0.04235722, -0.52815171,  1.15551598,
        -1.62749724,  0.71473237, -1.08458812,  4.66020296,  1.24563831,
        -0.67970862,  0.93461681,  1.18187607, -1.49501051,  2.44755622,
        -2.06424237, -0.04584074,  1.93396696,  1.07685273, -0.09837907], device=device)

for i in range(len(data)):
    p = p - 0.5 * torch.pow(apb - data[i], 2) 

exp_p = torch.exp(p - torch.max(p))
z = torch.sum(exp_p)
a_posterior = torch.sum(exp_p, 0) / z
b_posterior = torch.sum(exp_p, 1) / z

########################

end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end) * MS_TO_S)
#torch.set_printoptions(precision=10)
#print(a_posterior)
#print(b_posterior)
