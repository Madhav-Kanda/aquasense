import torch, time
import sys
import torch.distributions as tdist

MS_TO_S = 1/1000
aqua_device = torch.device("cuda:0")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ma_eps = 1.0E-9
start.record()
posterior_beta = [None] * 1
all_gt_beta = [None] * 1
lowProb_beta = [None] * 1
adaptiveLower_beta = [None] * 1
adaptiveUpper_beta = [None] * 1
repeat = True
adaptiveLower_sigma = ma_eps #10.0001
adaptiveUpper_sigma = 50 #0.001 #12 #50
adaptiveLower_beta[0] = -50 # 18 #-50
adaptiveUpper_beta[0] = 50 # 38 #50
N = 40
eps = float(sys.argv[2])
y = torch.tensor([12.1374259163952, 26.6903103048018, 38.5878897957254, 30.4930667829189, 39.2801876119107, 20.4174324499166, 25.7777563431966, 11.5919826316299, 37.3521894576722, 42.3729512347165, 33.1974931235148, 18.3925621724044, 38.2093756620254, 26.5178923853568, 4.52268843482463, 17.4645068462391, 20.0241067156045, 14.6755540100198, 12.9841025634912, 42.3191772083172, 29.558684814201, 30.911111042245, 25.211786124301, 22.3592414735694, 30.7775798797013, 16.8625137818992, 28.5736618568681, 19.5770593955495, 48.3010147870319, 31.2281915401234, 21.601456892799, 29.6730840465467, 30.9093803330748, 17.5249474584363, 37.2377810832606, 14.6787881367532, 24.770796317115, 34.6698348225234, 30.3842636610215, 24.5081046666978], device=aqua_device)
y = torch.reshape(y, [-1, 1, 1, 1, 1])
for eps_i in range(len(y)):
    y[eps_i] += eps
    
    while True:
        if repeat:
            splits = 100 #10
        else:
            splits = int(sys.argv[1])
        densityCube_p = torch.zeros(1, device=aqua_device)
        beta = [None] * 1
        beta[0] = torch.arange(adaptiveLower_beta[0], adaptiveUpper_beta[0] + ma_eps, step=(adaptiveUpper_beta[0] - adaptiveLower_beta[0])/splits, device=aqua_device)
        beta[0] = torch.reshape(beta[0], [1, -1, 1, 1, 1])
        sigma = torch.arange(adaptiveLower_sigma, adaptiveUpper_sigma + ma_eps, step=(adaptiveUpper_sigma - adaptiveLower_sigma)/splits, device=aqua_device)
        sigma = torch.reshape(sigma, [1, 1, -1, 1, 1])
        robust_local_nu = torch.arange(ma_eps + 2, 10 + ma_eps, step=(10 - 2 - ma_eps)/splits, device=aqua_device)
        robust_local_nu = torch.reshape(robust_local_nu, [1, 1, 1, -1, 1])
        robust_local_tau = [None] * 40
        for i in range(1, N + 1):
            robust_local_tau[i-1] = torch.arange(ma_eps, 10, step=(10 - ma_eps)/splits, device=aqua_device)
            robust_local_tau[i-1] = torch.reshape(robust_local_tau[i-1], [1, 1, 1, 1, -1])
            densityCube_p = densityCube_p + tdist.Gamma(robust_local_nu/2, (robust_local_nu/2)).log_prob(robust_local_tau[i-1]).to(aqua_device)
            #densityCube_p = densityCube_p + tdist.Gamma(3,2).log_prob(robust_local_tau[i-1]).to(aqua_device)
            densityCube_p = densityCube_p + (-torch.log(sigma*(1 / (torch.sqrt(robust_local_tau[i-1]))))- 0.9189385332046727 - 0.5 * torch.pow((beta[0] - (y.view(-1)[i-1])) / (sigma*(1 / (torch.sqrt(robust_local_tau[i-1])))), 2))
            densityCube_p = torch.logsumexp(densityCube_p, 4, keepdim=True)
    
        expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
        z_expDensityCube = torch.sum(expDensityCube_p)
        posterior_sigma = expDensityCube_p.sum([1, 0, 3, 4]) / z_expDensityCube
        posterior_beta[0] = expDensityCube_p.sum([0, 2, 3, 4]) / z_expDensityCube
        posterior_robust_local_nu = expDensityCube_p.sum([1, 2, 0, 4]) / z_expDensityCube
        sigma = sigma.flatten()
        posterior_sigma = posterior_sigma.flatten()
        beta[0] = beta[0].flatten()
        posterior_beta[0] = posterior_beta[0].flatten()
        robust_local_nu = robust_local_nu.flatten()
        posterior_robust_local_nu = posterior_robust_local_nu.flatten()
    
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
    
    end.record()
    #print((sigma.flatten() * posterior_sigma).sum())
    print((beta[0].flatten() * posterior_beta[0]).sum().item())
    #print((robust_local_nu.flatten() * posterior_robust_local_nu).sum())
    #print(start.elapsed_time(end) * MS_TO_S)
    #torch.set_printoptions(precision=8, sci_mode=False)
    #print(sigma.flatten())
    #print(posterior_sigma)
    #print(beta[0].flatten())
    #print(posterior_beta[0])
    y[eps_i] -= eps
