import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def compute_hvp(agent, sample, vector, batch_size):
    """Compute a Hessian-vector product."""

    p = len(parameters_to_vector(agent.policy_net.parameters()))
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    
    loss = agent.train(batch_size, *agent.process_sample(sample))
    grads = torch.autograd.grad(loss, inputs=agent.policy_net.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, agent.policy_net.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads)
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(agent, sample, batch_size, neigs=5):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(agent, sample, delta, batch_size).detach().cpu()
    theta_t = parameters_to_vector(agent.policy_net.parameters())
    nparams = len(theta_t)
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    scalar_product=  np.dot(evecs[:, 0], theta_t.detach().cpu().numpy())
    return (evals, scalar_product)