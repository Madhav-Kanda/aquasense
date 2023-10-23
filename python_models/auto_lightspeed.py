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

    def init_data(self):
        pass
        N = 40
        y = torch.tensor([12.1374259163952, 26.6903103048018, 38.5878897957254, 30.4930667829189, 39.2801876119107, 20.4174324499166, 25.7777563431966, 11.5919826316299, 37.3521894576722, 42.3729512347165, 33.1974931235148, 18.3925621724044, 38.2093756620254, 26.5178923853568, 4.52268843482463, 17.4645068462391, 20.0241067156045, 14.6755540100198, 12.9841025634912, 42.3191772083172, 29.558684814201, 30.911111042245, 25.211786124301, 22.3592414735694, 30.7775798797013, 16.8625137818992, 28.5736618568681, 19.5770593955495, 48.3010147870319, 31.2281915401234, 21.601456892799, 29.6730840465467, 30.9093803330748, 17.5249474584363, 37.2377810832606, 14.6787881367532, 24.770796317115, 34.6698348225234, 30.3842636610215, 24.5081046666978], device=self.device)
        y = torch.reshape(y, [-1, 1, 1])
        self.y = y
        self.N = N

    def analysis(self,adaptive_bounds):
        splits = self.splits
        beta = [None] * 1
        posterior_beta = [None] * 1
        adaptiveUpper_beta = [None] * 1
        adaptiveLower_beta = [None] * 1
        adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0], = adaptive_bounds
        y = self.y
        y = self.y + self.delta
        N = self.N
        ma_eps = 1.0E-9
        densityCube_p = torch.zeros(1, device=self.device)
        beta[0] = torch.arange(adaptiveLower_beta[0], adaptiveUpper_beta[0] + ma_eps, step=(adaptiveUpper_beta[0] - adaptiveLower_beta[0])/splits, device=self.device)
        beta[0] = torch.reshape(beta[0], [1, -1, 1])
        sigma = torch.arange(adaptiveLower_sigma, adaptiveUpper_sigma + ma_eps, step=(adaptiveUpper_sigma - adaptiveLower_sigma)/splits, device=self.device)
        sigma = torch.reshape(sigma, [1, 1, -1])
        densityCube_p = densityCube_p + torch.sum(torch.nan_to_num(-torch.log(sigma)- 0.9189385332046727 - 0.5 * torch.pow((beta[0] - (y)) / (sigma), 2), nan=-float('inf')), 0, keepdim=True)

        expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p))
        z_expDensityCube = torch.sum(expDensityCube_p)
        posterior_sigma = expDensityCube_p.sum([1, 0]) / z_expDensityCube
        posterior_beta[0] = expDensityCube_p.sum([0, 2]) / z_expDensityCube
        sigma = sigma.flatten()
        posterior_sigma = posterior_sigma.flatten()
        beta[0] = beta[0].flatten()
        posterior_beta[0] = posterior_beta[0].flatten()
        return [sigma,posterior_sigma,beta[0],posterior_beta[0]]

    def find_bounds(self):
        beta = [None] * 1
        posterior_beta = [None] * 1
        all_gt_beta = [None] * 1
        lowProb_beta = [None] * 1
        adaptiveLower_beta = [None] * 1
        adaptiveUpper_beta = [None] * 1
        repeat = True
        adaptiveLower_sigma = 1.0E-4
        adaptiveUpper_sigma = 50
        adaptiveLower_beta[0] = -50
        adaptiveUpper_beta[0] = 50
        while repeat:
            repeat = False
            sigma,posterior_sigma,beta[0],posterior_beta[0] = self.analysis([adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0],])
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
        return [adaptiveUpper_sigma,adaptiveLower_sigma,adaptiveUpper_beta[0],adaptiveLower_beta[0],]

    def run_analysis(self):
        self.init_data()
        self.delta = torch.tensor([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                    5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                            5., 5., 5., 5.], device=self.device).reshape(self.y.shape).requires_grad_(True)
        with torch.no_grad():
            bounds = self.find_bounds()
        beta = [None] * 1
        posterior_beta = [None] * 1
        expectation_beta = [None] * 1
        sigma,posterior_sigma,beta[0],posterior_beta[0] = self.analysis(bounds)
        if self.query == "sigma":
            print("sigma", (sigma.flatten() * posterior_sigma).sum().item())
            self.delta.grad = None
            expectation_sigma = (sigma.flatten() * posterior_sigma).sum()
            expectation_sigma.backward()
            print("dE[sigma]/dy", self.delta.grad.flatten())
        if self.query == "beta[0]":
            print("change beta[0]", (beta[0].flatten() * posterior_beta[0]).sum().item() - 26.55815887451172)
            self.delta.grad = None
            expectation_beta[0] = (beta[0].flatten() * posterior_beta[0]).sum()
            expectation_beta[0].backward()
            print("dE[beta[0]]/dy", self.delta.grad.flatten())
        MS_TO_S = 1/1000

ana = Analysis() 
ana.run_analysis()
