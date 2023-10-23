import torch
import sys
import torch.distributions as tdist

torch.set_printoptions(precision=8, sci_mode=False)
torch.distributions.Distribution.set_default_validate_args(False)

class Analysis:
    def __init__(self):
        self.splits = int(sys.argv[1])
        if sys.argv[2] == "GPU":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.query = sys.argv[3]
        self.beta = [None] * 2 #

    def init_data(self):
        pass
        N = 15
        y = torch.tensor([ 5.57649838108011, 7.7437901545178, 3.97144535706026, 6.90655408345038, 5.34519996931202, 5.82326082407921, 4.15702108539003, 6.81140925182179, 7.24764851401942, 5.49343534916886, 8.28785318207118, 6.7638307279417, 4.90294078296499, 5.66570297954388, 5.20315542517743], device=self.device)
        y_lag = torch.tensor([5.36950460318476, 9.45933439927176, 4.04464610107243, 6.75704792523757, 4.24326258972287, 6.9590606178157, 3.89350344482809, 5.34843717515469, 8.38955592149869, 5.99861560808495, 8.28150384919718, 7.6049918634817, 5.4332405702211, 5.35873385947198, 5.18218877464533], device=self.device)
        y = torch.reshape(y, [-1, 1, 1, 1])
        y_lag = torch.reshape(y_lag, [-1, 1, 1, 1])
        self.y = y
        self.y_lag = y_lag
        self.N = N

    def analysis_can_reuse(self, bounds, reuse):
        if reuse:
            return self.analysis_reuse()
        else:
            return self.analysis(bounds, add_delta=True)

    def analysis(self,adaptive_bounds, add_delta=False):
        splits = self.splits
        beta = [None] * 2
        posterior_beta = [None] * 2
        adaptiveUpper_beta = [None] * 2
        adaptiveLower_beta = [None] * 2
        adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0],adaptiveUpper_beta[1],adaptiveLower_beta[1], = adaptive_bounds
        y_lag = self.y_lag
        if add_delta:
            y = self.y + self.delta
        else:
            y = self.y
        N = self.N
        ma_eps = 1.0E-9
        densityCube_p = torch.zeros(1, device=self.device)
        beta[0] = torch.arange(adaptiveLower_beta[0], adaptiveUpper_beta[0] + ma_eps, step=(adaptiveUpper_beta[0] - adaptiveLower_beta[0])/splits, device=self.device)
        beta[0] = torch.reshape(beta[0], [1, -1, 1, 1])
        beta[1] = torch.arange(adaptiveLower_beta[1], adaptiveUpper_beta[1] + ma_eps, step=(adaptiveUpper_beta[1] - adaptiveLower_beta[1])/splits, device=self.device)
        beta[1] = torch.reshape(beta[1], [1, 1, -1, 1])
        sigma = torch.arange(adaptiveLower_sigma, adaptiveUpper_sigma + ma_eps, step=(adaptiveUpper_sigma - adaptiveLower_sigma)/splits, device=self.device)
        sigma = torch.reshape(sigma, [1, 1, 1, -1])
        densityCube_p = densityCube_p + torch.sum(torch.nan_to_num(-torch.log(sigma)- 0.9189385332046727 - 0.5 * torch.pow((beta[0]+beta[1]*y_lag - (y)) / (sigma), 2), nan=-float('inf')), 0, keepdim=True)

        self.densityCube_p = densityCube_p.detach() #.clone()
        self.beta[0] = beta[0] #
        self.beta[1] = beta[1] #
        self.sigma = sigma #

        expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p).item())
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
        return [sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1]]


    def analysis_reuse(self):
        beta = [None] * 2
        posterior_beta = [None] * 2
        beta = self.beta
        sigma = self.sigma
        densityCube_p = self.densityCube_p #
        delta = self.delta
        prev_delta = self.prev_delta
        mask = torch.abs(delta - prev_delta).ge(1.0E-5).flatten()
        y = self.y[mask]
        y_lag = self.y_lag[mask]
        densityCube_p = densityCube_p\
                - torch.sum(torch.nan_to_num(-torch.log(sigma)- 0.9189385332046727 - 0.5 * torch.pow((beta[0]+beta[1]*y_lag - (y + prev_delta[mask])) / (sigma), 2), nan=-float('inf')),0,keepdim=True) \
                + torch.sum(torch.nan_to_num(-torch.log(sigma)- 0.9189385332046727 - 0.5 * torch.pow((beta[0]+beta[1]*y_lag - (y + delta[mask])) / (sigma), 2), nan=-float('inf')), 0, keepdim=True)
        self.densityCube_p = densityCube_p.detach() #.clone()
        expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p).item())
        z_expDensityCube = torch.sum(expDensityCube_p)
        posterior_sigma = expDensityCube_p.sum([1, 2, 0]) / z_expDensityCube
        posterior_beta[0] = expDensityCube_p.sum([0, 2, 3]) / z_expDensityCube
        posterior_beta[1] = expDensityCube_p.sum([1, 0, 3]) / z_expDensityCube
        return [sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1]]


    def find_bounds(self):
        beta = [None] * 2
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
        adaptiveLower_beta[1] = -5
        adaptiveUpper_beta[1] = 5
        while repeat:
            repeat = False
            sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1] = self.analysis([adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0],adaptiveUpper_beta[1],adaptiveLower_beta[1],])
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
        return [adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0],adaptiveUpper_beta[1],adaptiveLower_beta[1],]

    def find_attack(self, bounds):
        beta = [None] * 2
        posterior_beta = [None] * 2
        expectation_beta = [None] * 2
        sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1] = self.analysis(bounds)
        old_beta = (beta[0].flatten() * posterior_beta[0].flatten()).sum().item()
        print("beta[0]", (beta[0].flatten() * posterior_beta[0].flatten()).sum().item())
        eps = 1 #
        lr = 1 #

        with torch.no_grad():
            self.delta = (torch.randn_like(self.y, device=self.device) / 100)
            self.delta.flatten()[3:] = 0
            self.delta.clamp_(min=-eps, max=eps) #.clone().detach().requires_grad_(True)
            self.delta.requires_grad_(True)
        for k in range(100):
            sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1] = self.analysis_can_reuse(bounds, True)
            expectation_beta[0] = (beta[0].flatten() * posterior_beta[0].flatten()).sum()
            diff = torch.abs(expectation_beta[0] - old_beta)
            diff.backward()
            with torch.no_grad():
                grad = self.delta.grad 
                self.prev_delta = self.delta.clone().detach() #
                self.delta += lr * grad.sign()
                self.delta.grad.zero_()
                self.delta.flatten()[3:] = 0
                self.delta.clamp_(min=-eps, max=eps) #.clone().detach().requires_grad_(True)
            if (torch.all(grad < 1E-5)):
                break

    def run_analysis(self):
        self.init_data()
        self.delta = torch.zeros_like(self.y, device=self.device, requires_grad=True)
        self.prev_delta = torch.zeros_like(self.y, device=self.device) #
        beta = [None] * 2
        posterior_beta = [None] * 2
        with torch.no_grad():
            bounds = self.find_bounds()
        self.find_attack(bounds)
        with torch.no_grad():
            sigma,posterior_sigma,beta[0],posterior_beta[0],beta[1],posterior_beta[1] = self.analysis_can_reuse(bounds,True)
        print(self.delta.flatten())
        print(posterior_beta[1])
        print("beta[0]", (beta[0].flatten() * posterior_beta[0].flatten()).sum().item())
        MS_TO_S = 1/1000

ana = Analysis() 
ana.run_analysis()
