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


torch.manual_seed(123)
d, w0 = 2, 20
print(f"True value of mu: {2*d}")

t_obs = torch.rand(40, device=device).view(-1, 1)
u_obs = solution(d, w0, t_obs) + 0.04 * torch.randn_like(t_obs)

plt.figure()
plt.title("Noisy observational data")
plt.scatter(t_obs.cpu()[:, 0], u_obs.cpu()[:, 0])
t_test = torch.linspace(0, 1, 300, device=device).view(-1, 1)
u_exact = solution(d, w0, t_test)
plt.plot(t_test.cpu()[:, 0], u_exact.cpu()[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
plt.legend()
plt.savefig("noisy_observed_data.png")
plt.show()
plt.close()


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


torch.manual_seed(123)

pinn = NaiveNeuralNetwork(1, 1, 32, 3).to(device)
t_physics = torch.linspace(0, 1, 30, device=device).view(-1, 1).requires_grad_(True)

d, w0 = 2, 20
_, k = 2 * d, w0**2

mu = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
mus = []

optimizer = torch.optim.Adam(list(pinn.parameters()) + [mu], lr=1e-3)

for i in range(15001):
    optimizer.zero_grad()

    lambda1 = 1e4
    u = pinn(t_physics)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss1 = torch.mean((d2udt2 + mu * dudt + k * u)**2)

    u = pinn(t_obs)
    loss2 = torch.mean((u - u_obs)**2)

    loss = loss1 + lambda1 * loss2
    loss.backward()
    optimizer.step()

    mus.append(mu.item())

    if i % 5000 == 0:
        print(f"Step {i}, Mu: {mu.item():.4f}, Loss1: {loss1.item():.6f}, Loss2: {loss2.item():.6f}")
        
        u = pinn(t_test).detach().cpu()
        
        plt.figure(figsize=(6, 2.5))
        plt.scatter(t_obs.cpu()[:, 0], u_obs.cpu()[:, 0], label="Noisy observations", alpha=0.6)
        plt.plot(t_test.cpu()[:, 0], u[:, 0], label="PINN solution", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.savefig(f"inverse_problem_step_{i}.png")
        plt.show()
        plt.close()

plt.figure()
plt.title("$\mu$")
plt.plot(mus, label="PINN estimate")
plt.hlines(2 * d, 0, len(mus), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.savefig("mu_convergence.png")
plt.show()
plt.close()