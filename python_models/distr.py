def flip(p, device, shape=None log=True):
    if not shape:
        single =  torch.tensor([0,1], device=device)
        prob = torch.log2(torch.tensor([1-p, p], device=device))
    else:
        single = torch.tensor([0,1], device=device).reshape(shape)
        prob = torch.log2(torch.tensor([1-p, p], device=device).reshape(shape))
    if log:
        return single, torch.log2(prob)
    else:
        return single, prob


