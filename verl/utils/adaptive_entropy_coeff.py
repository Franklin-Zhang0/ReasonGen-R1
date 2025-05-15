import torch

class AdaptiveEntropyCoefficient:
    """
    Learns an entropy coefficient that can go negative, parameterized via
    ψ on the real line, with α = sinh(ψ) (so α ∈ ℝ and |ψ| ≈ ln|α| for large α).
    """
    def __init__(self, initial_alpha=0.0, target_entropy=1.0, lr=1e-5, device="cpu", max_coeff=1e-3, min_coeff=-1e-3):
        # initialize ψ = arcsinh(initial_alpha)
        init_psi = initial_alpha  # if you want exact arcsinh: torch.asinh(torch.tensor(initial_alpha))
        self.psi = torch.nn.Parameter(torch.asinh(torch.tensor(init_psi, device=device)))
        self.target_entropy = target_entropy
        self.opt = torch.optim.Adam([self.psi], lr=lr)
        self.max_coeff = max_coeff
        self.min_coeff = min_coeff
    
    @property
    def alpha(self):
        # α = sinh(ψ), can be negative
        alpha = torch.sinh(self.psi)
        return alpha
    
    def get_alpha(self):
        # α = sinh(ψ), can be negative
        alpha = torch.sinh(self.psi)
        alpha = torch.clamp(alpha, min=self.min_coeff, max=self.max_coeff)
        return alpha.detach()
    
    def update(self, entropy):
        ent = entropy.detach()
        # loss = -α * (ent - target)  (so pushes α > 0 when ent < target, α < 0 when ent > target)
        loss = - (self.alpha * (ent - self.target_entropy)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # compute psi based on clipped alpha
        self.psi.data = torch.asinh(self.get_alpha())
        return loss.item()

