import torch, time
import sys
import torch.distributions as tdist

MS_TO_S = 1/1000
aqua_device = torch.device("cuda:0")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ma_eps = 1.0E-9
start.record()
posterior_beta = [None] * 2
all_gt_beta = [None] * 2
lowProb_beta = [None] * 2
adaptiveLower_beta = [None] * 2
adaptiveUpper_beta = [None] * 2
repeat = True
adaptiveLower_sigma = 1.0E-4
adaptiveUpper_sigma = 50
adaptiveLower_beta[0] = -50
adaptiveUpper_beta[0] = 50
adaptiveLower_beta[1] = -5 #-50
adaptiveUpper_beta[1] = 5 #50
N = 40
y = torch.tensor([3.33008960302557, 5.19543854472405, 5.88929762709886, 5.52449264517973, 5.31172037906861, 7.1453284527505, 7.11693949967702, 10.2790569556659, 8.70290237657601, 4.91879758555161, 5.9793649285972, 5.71069265888431, 5.82376740342438, 6.63877402512103, 7.42179960481787, 9.62291014033926, 2.3662166776056, 5.13435595049398, 8.88839345256223, 4.82787281003398, 6.66222539318123, 5.52684066394334, 5.1114875649346, 5.63288986028277, 4.38694756020315, 5.57649838108011, 7.7437901545178, 3.97144535706026, 6.90655408345038, 5.34519996931202, 5.82326082407921, 4.15702108539003, 6.81140925182179, 7.24764851401942, 5.49343534916886, 8.28785318207118, 6.7638307279417, 4.90294078296499, 5.66570297954388, 5.20315542517743], device=aqua_device)
y = torch.reshape(y, [-1, 1, 1, 1])
y_lag = torch.tensor([4.3838241041638, 3.93442675489932, 7.57050890065729, 4.53683034032583, 5.28768584504724, 7.84145292649045, 8.09962392030284, 9.55146255046129, 8.73574461648241, 4.44520985772833, 4.86994492644444, 4.09735724627972, 4.01458069570362, 8.93653732435778, 6.37760733487085, 9.47473778631538, 3.34918157150969, 5.00719783334061, 8.86413662843406, 4.86521017467603, 6.06770903747529, 6.16693980395794, 7.25456838915125, 5.95538431135938, 5.22133663948625, 5.36950460318476, 9.45933439927176, 4.04464610107243, 6.75704792523757, 4.24326258972287, 6.9590606178157, 3.89350344482809, 5.34843717515469, 8.38955592149869, 5.99861560808495, 8.28150384919718, 7.6049918634817, 5.4332405702211, 5.35873385947198, 5.18218877464533], device=aqua_device)
y_lag = torch.reshape(y_lag, [-1, 1, 1, 1])

eps = float(sys.argv[2])

for eps_i in range(len(y)):
    y[eps_i] += eps


    while True:
        if repeat:
            splits = int(sys.argv[1])
        else:
            splits = int(sys.argv[1])
        densityCube_p = torch.zeros(1, device=aqua_device)
        beta = [None] * 2
        beta[0] = torch.arange(adaptiveLower_beta[0], adaptiveUpper_beta[0] + ma_eps, step=(adaptiveUpper_beta[0] - adaptiveLower_beta[0])/splits, device=aqua_device)
        beta[0] = torch.reshape(beta[0], [1, -1, 1, 1])
        beta[1] = torch.arange(adaptiveLower_beta[1], adaptiveUpper_beta[1] + ma_eps, step=(adaptiveUpper_beta[1] - adaptiveLower_beta[1])/splits, device=aqua_device)
        beta[1] = torch.reshape(beta[1], [1, 1, -1, 1])
        sigma = torch.arange(adaptiveLower_sigma, adaptiveUpper_sigma + ma_eps, step=(adaptiveUpper_sigma - adaptiveLower_sigma)/splits, device=aqua_device)
        sigma = torch.reshape(sigma, [1, 1, 1, -1])
        densityCube_p = densityCube_p + torch.sum((-torch.log(sigma)- 0.9189385332046727 - 0.5 * torch.pow((beta[0]+beta[1]*y_lag - (y)) / (sigma), 2)), 0, keepdim=True)

        expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
        z_expDensityCube = torch.sum(expDensityCube_p)
        posterior_sigma = expDensityCube_p.sum([1, 2, 0]) / z_expDensityCube
        posterior_beta[0] = expDensityCube_p.sum([0, 2, 3]) / z_expDensityCube
        posterior_beta[1] = expDensityCube_p.sum([1, 0, 3]) / z_expDensityCube
        sigma = sigma.flatten()
        posterior_sigma = posterior_sigma.flatten()
        beta[0] = beta[0].flatten()
        posterior_beta[0] = posterior_beta[0].flatten()
        beta[1] = beta[1].flatten()
        posterior_beta[1] = posterior_beta[1].flatten()

        if repeat == False:
            break

        repeat = False
        lowProb_sigma = posterior_sigma.max() * 0.001
        all_gt_sigma = (posterior_sigma > lowProb_sigma).nonzero(as_tuple=True)[0]
        if abs(all_gt_sigma[0] - all_gt_sigma[-1]) < 2 and not ((sigma[max(all_gt_sigma[0] - 1, 0)] == adaptiveLower_sigma) or (sigma[min(all_gt_sigma[-1] + 1, len(sigma) - 1)] == adaptiveUpper_sigma)):
            repeat = True
        adaptiveLower_sigma = sigma[max(all_gt_sigma[0] - 1, 0)]
        adaptiveUpper_sigma = sigma[min(all_gt_sigma[-1] + 1, len(sigma) - 1)]

        lowProb_beta[0] = posterior_beta[0].max() * 0.001
        all_gt_beta[0] = (posterior_beta[0] > lowProb_beta[0]).nonzero(as_tuple=True)[0]
        if abs(all_gt_beta[0][0] - all_gt_beta[0][-1]) < 2 and not ((beta[0][max(all_gt_beta[0][0] - 1, 0)] == adaptiveLower_beta[0]) or (beta[0][min(all_gt_beta[0][-1] + 1, len(beta[0]) - 1)] == adaptiveUpper_beta[0])):
            repeat = True
        adaptiveLower_beta[0] = beta[0][max(all_gt_beta[0][0] - 1, 0)]
        adaptiveUpper_beta[0] = beta[0][min(all_gt_beta[0][-1] + 1, len(beta[0]) - 1)]

        lowProb_beta[1] = posterior_beta[1].max() * 0.001
        all_gt_beta[1] = (posterior_beta[1] > lowProb_beta[1]).nonzero(as_tuple=True)[0]
        if abs(all_gt_beta[1][0] - all_gt_beta[1][-1]) < 2 and not ((beta[1][max(all_gt_beta[1][0] - 1, 0)] == adaptiveLower_beta[1]) or (beta[1][min(all_gt_beta[1][-1] + 1, len(beta[1]) - 1)] == adaptiveUpper_beta[1])):
            repeat = True
        adaptiveLower_beta[1] = beta[1][max(all_gt_beta[1][0] - 1, 0)]
        adaptiveUpper_beta[1] = beta[1][min(all_gt_beta[1][-1] + 1, len(beta[1]) - 1)]

    y[eps_i] -= eps
    #print((sigma.flatten() * posterior_sigma).sum().item())
    #print(str(y[eps_i].item()) + "," + str((beta[0].flatten() * posterior_beta[0]).sum().item()))
    #print((beta[0].flatten() * posterior_beta[0]).sum().item())
    #print((beta[1].flatten() * posterior_beta[1]).sum().item())
    print(str(y[eps_i].item()) + "," + str((beta[1].flatten() * posterior_beta[1]).sum().item()))

end.record()
#print(start.elapsed_time(end) * MS_TO_S)
