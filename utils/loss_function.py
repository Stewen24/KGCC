from typing import Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
## implementation of domain adversarial traning. For more details, please visit: https://dalib.readthedocs.io/en/latest/index.html
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module,reduction: Optional[str] = 'mean',max_iter=1000):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))

class CustomLoss:
    def __init__(self, dann_loss, hidden_4, eta=1e-5,device="cuda"):
        self.dann_loss = dann_loss
        self.hidden_4 = hidden_4
        self.eta = eta
        self.device = device

    def __call__(self, model, boost_factor, feature_source_f, feature_target_f, sim_matrix, sim_matrix_target, 
                 estimated_sim_truth, estimated_sim_truth_target, batch_size):
        # 分类损失
        bce_loss = -(torch.log(sim_matrix + self.eta) * estimated_sim_truth) - \
                   (1 - estimated_sim_truth) * torch.log(1 - sim_matrix + self.eta)
        bce_loss_target = -(torch.log(sim_matrix_target + self.eta) * estimated_sim_truth_target) - \
                          (1 - estimated_sim_truth_target) * torch.log(1 - sim_matrix_target + self.eta)
        cls_loss = torch.mean(bce_loss)

        # 聚类损失
        indicator, nb_selected = model.compute_indicator(sim_matrix_target)
        cluster_loss = torch.sum(indicator * bce_loss_target) / nb_selected

        # 正则化损失
        P_loss = torch.norm(torch.matmul(model.P.T, model.P) - torch.eye(self.hidden_4).to(self.device), 'fro')

        # 对抗损失
        transfer_loss = self.dann_loss(
            feature_source_f + 0.005 * torch.randn((batch_size, self.hidden_4)).to(self.device),
            feature_target_f + 0.005 * torch.randn((batch_size, self.hidden_4)).to(self.device)
        )

        # 总损失
        total_loss = cls_loss + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss

        return total_loss, cls_loss, transfer_loss

