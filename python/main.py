from sympy import Matrix, symbols, sin, sqrt, latex
from sympy.printing import ccode

# Define coordinate symbols (x_t x_r x_theta x_phi) and rs (Schwarzschild Radius)
x_t, x_r, x_theta, x_phi = symbols('x_t x_r x_theta x_phi', real=True)
coords = [x_t, x_r, x_theta, x_phi]

# Define constants depended on by the Metric
rs = symbols('rs', real=True)

# Define Metric in matrix form
g = Matrix([
    [-(1 - rs / x_r), 0, 0, 0],
    [0, 1 / (1 - rs / x_r), 0, 0],
    [0, 0, x_r ** 2, 0],
    [0, 0, 0, x_r ** 2 * sin(x_theta) ** 2],
])

# Find inverse metric
g_inv = g.inv()

# Symbolically find the tetrad
e = Matrix.zeros(4, 4)
for mu in range(4):
    e[mu, mu] = 1 / sqrt(g[mu, mu] * (-1 if mu == 0 else 1))

# Define p (momentum) for the Hamiltonian
p_t, p_r, p_theta, p_phi = symbols("p_t p_r p_theta p_phi", real=True)
p = Matrix([p_t, p_r, p_theta, p_phi])

# Symbolic expression for the Hamiltonian for the geodesic
H = (p.T * g_inv * p)[0] / 2

# Hamilton's equations
dx = [H.diff(p_mu) for p_mu in p]           # dx^μ/dλ = ∂H/∂p_μ
dp = [-H.diff(coord) for coord in coords]   # dp_μ/dλ = -∂H/∂x^μ

print("")
print("----------------------- RESULT -----------------------")

print("Metric (g):", g)
print("Inverse Metric (g^-1):", g_inv)
print("H(x, p):", H)
print("Tetrad:", e)

print("")
print("--- ODE ---")

print("ODE's derived from the Hamiltonian of the geodesic:")

# dx^μ/dλ = ∂H/∂p_μ
for i, expr in enumerate(dx):
    print(f"dx[{i}]/dλ = {ccode(expr)};")

print("")

# dp_μ/dλ = -∂H/∂x^μ
for i, expr in enumerate(dp):
    print(f"dp[{i}]/dλ = {ccode(expr)};")

print("")
print("--- LATEX ---")

# dx^μ/dλ = ∂H/∂p_μ
for i, expr in enumerate(dx):
    print(r"\frac{dx_{" + str(coords[i]) + r"}}{d\lambda}=" + latex(expr) + r"\\")

# dp_μ/dλ = -∂H/∂x^μ
for i, expr in enumerate(dp):
    print(r"\frac{dp_{" + str(coords[i]) + r"}}{d\lambda}=" + latex(expr) + r"\\")
