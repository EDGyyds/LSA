import torch
from utils.utils import maybe_cuda
import torch.nn.functional as F
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
from copy import deepcopy
import numpy as np
import copy
# from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class LSA_I(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.num_retrieve = params.eps_mem_batch
        self.return_indices = False
        self.mem_size = params.mem_size
        self.subsample = params.subsample
        self.beta = params.beta
    
    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = buffer.model.forward(sub_x)
                logits_numpy = logits_pre.cpu().numpy()
                dist = cdist(logits_numpy,logits_numpy, metric='euclidean')
                # Calculate distribution-diversity score by using Eq(7)
                diversity = np.sum(dist,1) / np.sum(dist)
                diversity = maybe_cuda(torch.from_numpy(diversity))
                logits_post = model_temp.forward(sub_x)               
                # Calculate stream-sensitivity score by using Eq(6)
                informative = torch.norm(logits_post - logits_pre, p = 1, dim = 1) / torch.sum(torch.norm(logits_post - logits_pre, p = 1, dim = 1))
                # Obtain replay sample set by using Eq(9), self.beta is the trade-off weight
                big_ind = (informative + self.beta * diversity).sort(descending=True)[1][:self.num_retrieve]

            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y
       
    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
         






