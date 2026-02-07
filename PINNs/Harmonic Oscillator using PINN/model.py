import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def solution(delta, w0, t):
    if not isinstance(delta, torch.Tensor):
        delta = torch.tensor(delta, dtype=t.dtype, device=t.device)
    if not isinstance(w0, torch.Tensor):
        w0 = torch.tensor(w0, dtype=t.dtype, device=t.device)
    
    assert delta < w0
    w = torch.sqrt(w0**2 - delta**2)
    exp = torch.exp(-delta * t)
    phi = torch.arctan(-delta / w)
    A = 1 / (2 * torch.cos(phi))
    cos = torch.cos(phi + w * t)
    u = exp * 2 * A * cos
    return u


class NaiveNeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, num_layers):
        super().__init__()
        activation = nn.Tanh
        
        self.input_layer = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            activation()
        )

        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                activation()
            ) for _ in range(num_layers - 1)
        ])

        self.output_layer = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


torch.manual_seed(1235)

pinn = NaiveNeuralNetwork(1, 1, 32, 3).to(device)

t_boundary = torch.tensor(0., device=device).view(-1, 1).requires_grad_(True)
t_physics = torch.linspace(0, 1, 30, device=device).view(-1, 1).requires_grad_(True)

d, w0 = 2, 20
mu, k = 2 * d, w0**2

t_test = torch.linspace(0, 1, 300, device=device).view(-1, 1)
u_exact = solution(d, w0, t_test)

optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)


for i in range(15001):
    optimizer.zero_grad()
    lambda1, lambda2 = 1e-1, 1e-4

    # boundary loss (u(0) = 1, u'(0) = 0)
    u = pinn(t_boundary)
    loss1 = (torch.squeeze(u) - 1)**2
    
    dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - 0)**2

    # physics loss
    u = pinn(t_physics)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss3 = torch.mean((d2udt2 + mu * dudt + k * u)**2)

    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
    loss.backward()
    optimizer.step()

    if i % 5000 == 0:
        u = pinn(t_test).detach().cpu()
        
        plt.figure(figsize=(6, 2.5))
        plt.scatter(t_physics.detach().cpu()[:, 0],
                    torch.zeros_like(t_physics.cpu())[:, 0], s=20, lw=0, color="tab:green", alpha=0.6)
        plt.scatter(t_boundary.detach().cpu()[:, 0],
                    torch.zeros_like(t_boundary.cpu())[:, 0], s=20, lw=0, color="tab:red", alpha=0.6)
        plt.plot(t_test.cpu()[:, 0], u_exact.cpu()[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
        plt.plot(t_test.cpu()[:, 0], u[:, 0], label="PINN solution", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.savefig(f"plot_at_training_step_{i}.png")  
        plt.show()
        plt.close()  