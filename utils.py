class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = None
        self.count = None
        self.val = None
        self.avg = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums
