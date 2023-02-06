import torch
import torch.nn.functional as F


def toggle_bn_tracking_stats(layer):
    if hasattr(layer, 'track_running_stats'):
        layer.track_running_stats ^= True


def l2_normalize(x):
    x_reshaped = x.view(x.shape[0], -1, *(1 for _ in range(x.dim()-2)))
    l2_norm = torch.norm(x_reshaped, dim=1, keepdim=True)
    return x / (l2_norm + 1e-16)


class VAT:
    '''
    Virtual Adversarial Training

    !!! Limitation !!!
    This module cannot be used with nn.Dropout
    '''

    def __init__(self, epsilon=1.0, xi=1e-6, ip=1):
        self._epsilon = epsilon
        self._xi = xi
        self._ip = ip

    def forward(self, model, x):
        # stop tracking batch normalization stats
        model.apply(toggle_bn_tracking_stats)

        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)

        d_adv = self.get_adv_noise(model, x, prob)

        log_prob_adv = F.log_softmax(model(x+self._epsilon*d_adv), dim=1)
        loss = F.kl_div(log_prob_adv, prob, reduction='batchmean')

        # start tracking batch normalization stats
        model.apply(toggle_bn_tracking_stats)

        return loss

    def get_adv_noise(self, model, x, prob=None):
        d = torch.rand(x.shape, device=x.device)
        d = l2_normalize(d)

        for _ in range(self._ip):
            d.requires_grad_()
            log_prob_d = F.log_softmax(model(x + self._xi * d), dim=1)
            loss = F.kl_div(log_prob_d, prob, reduction='batchmean')
            loss.backward()
            d = l2_normalize(d.grad)
            model.zero_grad()

        d_adv = d.detach()

        return d_adv
