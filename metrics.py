# Sense
# distance metrics
from unittest import skip
import torch
import functools
import sys

ERR_CODE = -1

# Expectation Distance 1, see PSense paper for documentation
def ED1(x, px, x_noise, px_noise):
    return abs((x * px).sum().item() - (x_noise * px_noise).sum().item())

# Kolmogorov-Smirnov statistic
def KS(x, px, x_noise, px_noise):
    # truncate the supports of truth and eps, so that Supp_eps is strictly within Supp_truth
    if (x.tolist() == x_noise.tolist()):
        for i in range(1, len(px)):
            px[i] += px[i-1]
        for i in range(1, len(px_noise)):
            px_noise[i] += px_noise[i-1]
        ks = torch.max(torch.abs(px - px_noise)).item()

    else:
        x = x.tolist()
        px = px.tolist()

        x_noise = x_noise.tolist()
        px_noise = px_noise.tolist()

        while(len(x_noise) and x_noise[0] < x[0]):
            x_noise = x_noise[1:]
            px_noise = px_noise[1:]
        while (len(x_noise) and x_noise[-1] > x[-1]):
            x_noise = x_noise[:-1]
            px_noise = px_noise[:-1]
        
        ks = 0

        for i in range(1, len(px)):
            px[i] += px[i-1]
        for i in range(1, len(px_noise)):
            px_noise[i] += px_noise[i-1]
        
        # if support has no overlapping, KS is maxmized
        if (not len(x_noise)):
            return 1
        
        j = 0
        for i in range(len(x_noise)):
            while not (x[j] <= x_noise[i] and x_noise[i] <= x[j+1]):
                j += 1
            ks = max(ks, abs(px[j] - px_noise[i]))
    
    return ks

# A demo user distance metric
def user(x, px, x_noise, px_noise):    
    return abs((x * px).sum().item() - (x_noise * px_noise).sum().item())

# add the name of your customized distance metric to the dict
metr = {"expdist1" : ED1, "user" : user, "KS" : KS}