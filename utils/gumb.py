import torch


# Base on 
# https://pytorch.org/docs/stable/nn.html?highlight=gumbel_softmax#torch.nn.functional.gumbel_softmax
# modified torch.nn.functional.gumbel_softmax() 
# so now it works on 3D input [sample_size, batch_size, num_features]

def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    """
    if out is None:
        U = torch.rand(shape)
    else:
        U = torch.jit._unwrap_optional(out).resize_(shape).uniform_()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=torch.empty_like(logits))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y/tau, dims-1)


def gumbel_softmax_3d(logits, tau=1., hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[sample_size, batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``sample_size x batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features
      
    """
    shape = logits.size()
    assert len(shape) == 3 #[bs, sz, ncats]
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(shape, dtype=logits.dtype, 
                             device=logits.device).scatter_(-1, torch.unsqueeze(k, 2), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y
