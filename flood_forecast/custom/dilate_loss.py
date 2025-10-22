import numpy as np
import torch
from numba import jit
from torch.autograd import Function


class DilateLoss(torch.nn.Module):
    def __init__(self, gamma=0.001, alpha=0.5):
        """
        Dilate loss function originally from https://github.com/manjot4/NIPS-Reproducibility-Challenge.

        :param gamma: Parameter for the Soft-DTW calculation.
        :type gamma: float
        :param alpha: Weighting factor between the shape loss and the temporal loss.
        :type alpha: float
        :return: None
        :rtype: None
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, targets: torch.Tensor, outputs: torch.Tensor):
        """
        Computes the Dilate Loss between targets and outputs.

        :param targets: The ground truth tensor of time series.
         :type targets: torch.Tensor
         :param outputs: The predicted tensor of time series.
         :type outputs: torch.Tensor
          :return: The total Dilate loss, a weighted sum of shape and temporal losses.
          :rtype: torch.Tensor
        """
        outputs = outputs.float()
        targets = targets.float()
        # outputs, targets: shape (batch_size, N_output, 1)
        if len(targets.size()) < 2:
            print("begin fixed loss func")
            targets = targets.unsqueeze(0)
            outputs = outputs.unsqueeze(0)
        if len(targets.size()) < 3:
            outputs = outputs.unsqueeze(2)
            targets = targets.unsqueeze(2)
        batch_size, N_output = outputs.shape[0:2]
        loss_shape = 0
        softdtw_batch = SoftDTWBatch.apply
        D = torch.zeros((batch_size, N_output, N_output)).to(self.device)
        for k in range(batch_size):
            Dk = pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = softdtw_batch(D, self.gamma)
        path_dtw = PathDTWBatch.apply
        path = path_dtw(D, self.gamma)
        Omega = pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(self.device)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss


def pairwise_distances(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the square of the Euclidean distance between all pairs of vectors.

    :param x: A tensor of dimension (Nxd).
     :type x: torch.Tensor
     :param y: An optional tensor of dimension (Mxd). If None, y=x is used.
     :type y: torch.Tensor or None
      :return: A tensor of dimension (NxM) where dist[i,j] is the square norm between x[i,:] and y[j,:].
      :rtype: torch.Tensor
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


@jit(nopython=True)
def compute_softdtw(D: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the Soft Dynamic Time Warping (Soft-DTW) distance matrix.

    :param D: The pairwise distance matrix (cost matrix).
     :type D: np.ndarray
     :param gamma: The smoothing parameter for the soft minimum.
     :type gamma: float
      :return: The accumulated cost matrix R, where R[N, M] is the Soft-DTW distance.
      :rtype: np.ndarray
    """
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R


@jit(nopython=True)
def compute_softdtw_backward(D_: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the gradient of the Soft-DTW loss with respect to the cost matrix D (the E matrix).

    :param D_: The pairwise distance matrix (cost matrix).
     :type D_: np.ndarray
     :param R: The accumulated cost matrix from the forward pass.
     :type R: np.ndarray
     :param gamma: The smoothing parameter for the soft minimum.
     :type gamma: float
      :return: The E matrix, representing the gradient of the Soft-DTW with respect to D.
      :rtype: np.ndarray
    """
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:N + 1, 1:M + 1]


class SoftDTWBatch(Function):
    """
    Pytorch autograd Function for computing the Soft-DTW loss in a batch.
    """
    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        Performs the forward pass for Soft-DTW loss computation.

        :param D: The batch of pairwise distance matrices. D.shape: [batch_size, N , N]
         :type D: torch.Tensor
         :param gamma: The smoothing parameter for the soft minimum.
         :type gamma: float
          :return: The mean Soft-DTW loss across the batch.
          :rtype: torch.Tensor
        """
        dev = D.device
        batch_size, N, N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N + 2, N + 2)).to(dev)
        for k in range(0, batch_size):  # loop over all D in the batch
            Rk = torch.FloatTensor(compute_softdtw(D_[k, :, :], g_)).to(dev)
            R[k:k + 1, :, :] = Rk
            total_loss = total_loss + Rk[-2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> (torch.Tensor, None):
        """
        Performs the backward pass for Soft-DTW loss, computing the gradient w.r.t. D.

        :param grad_output: The gradient of the loss w.r.t. the output of the forward pass (total_loss / batch_size).
         :type grad_output: torch.Tensor
          :return: The gradient of the loss w.r.t. the input D, and None for gamma (since it's not a learnable parameter).
          :rtype: tuple of (torch.Tensor, None)
        """
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N, N)).to(dev)
        for k in range(batch_size):
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k, :, :], R_[k, :, :], g_)).to(dev)
            E[k:k + 1, :, :] = Ek

        return grad_output * E, None


@jit(nopython=True)
def my_max(x: np.ndarray, gamma: float) -> (float, np.ndarray):
    """
    Computes the soft maximum of an array using the log-sum-exp trick.

    :param x: The input array.
     :type x: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: A tuple containing the soft maximum value and the probability array (softmax/argmax).
      :rtype: tuple of (float, np.ndarray)
    """
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x: np.ndarray, gamma: float) -> (float, np.ndarray):
    """
    Computes the soft minimum of an array using the soft maximum on the negative array.

    :param x: The input array.
     :type x: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: A tuple containing the soft minimum value and the probability array (softmax/argmax of -x).
      :rtype: tuple of (float, np.ndarray)
    """
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x


@jit(nopython=True)
def my_max_hessian_product(p: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the Hessian-vector product for the soft maximum function.

    :param p: The probability array (softmax output) from the first derivative calculation.
     :type p: np.ndarray
     :param z: The input vector (derivative of the argument of the soft-max).
     :type z: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: The Hessian-vector product.
      :rtype: np.ndarray
    """
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the Hessian-vector product for the soft minimum function.

    :param p: The probability array (softmax output) from the first derivative calculation.
     :type p: np.ndarray
     :param z: The input vector (derivative of the argument of the soft-min).
     :type z: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: The Hessian-vector product.
      :rtype: np.ndarray
    """
    return - my_max_hessian_product(p, z, gamma)


@jit(nopython=True)
def dtw_grad(theta: np.ndarray, gamma: float) -> (float, np.ndarray, np.ndarray, np.ndarray):
    """
    Computes the Soft-DTW loss, its gradient (path), and intermediate matrices for the Hessian product.

    :param theta: The pairwise distance matrix D (cost matrix).
     :type theta: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: A tuple containing the Soft-DTW loss value (V[m, n]), the gradient (E[1:m+1, 1:n+1]), the probability matrix Q, and the full E matrix.
      :rtype: tuple of (float, np.ndarray, np.ndarray, np.ndarray)
    """
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                          V[i - 1, j - 1],
                                          V[i - 1, j]]), gamma)
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta: np.ndarray, Z: np.ndarray, Q: np.ndarray, E: np.ndarray, gamma: float) -> (float, np.ndarray):
    """
    Computes the Hessian-vector product of the Soft-DTW loss with respect to the cost matrix D.

    :param theta: The pairwise distance matrix D (cost matrix).
     :type theta: np.ndarray
     :param Z: The vector 'v' for the Hessian-vector product (e.g., the output gradient grad_output).
     :type Z: np.ndarray
     :param Q: The probability matrix from the dtw_grad forward pass.
     :type Q: np.ndarray
     :param E: The full E matrix (path/gradient) from the dtw_grad forward pass.
     :type E: np.ndarray
     :param gamma: The smoothing parameter.
     :type gamma: float
      :return: A tuple containing the final V_dot value and the Hessian-vector product matrix E_dot.
      :rtype: tuple of (float, np.ndarray)
    """
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                Q[i, j, 0] * V_dot[i, j - 1] + \
                Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    """
    Pytorch autograd Function for computing the Soft-DTW path (gradient) in a batch.
    This path is used as the temporal loss component in DilateLoss.
    """
    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Performs the forward pass for Soft-DTW path computation.

        :param D: The batch of pairwise distance matrices. D.shape: [batch_size, N , N]
         :type D: torch.Tensor
         :param gamma: The smoothing parameter for the soft minimum.
         :type gamma: float
          :return: The mean Soft-DTW path (gradient) across the batch.
          :rtype: torch.Tensor
        """
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, N + 2, N + 2)).to(device)

        for k in range(0, batch_size):  # loop over all D in the batch
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> (torch.Tensor, None):
        """
        Performs the backward pass for the Soft-DTW path, computing the Hessian-vector product w.r.t. D.

        :param grad_output: The gradient of the loss w.r.t. the output of the forward pass (mean path).
         :type grad_output: torch.Tensor
          :return: The Hessian-vector product (Hessian matrix of the Soft-DTW loss * grad_output) w.r.t. D, and None for gamma.
          :rtype: tuple of (torch.Tensor, None)
        """
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()

        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N)).to(device)
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma)
            Hessian[k:k + 1, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None