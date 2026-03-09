import math
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def vorst_algo(u, d, R, S0, K, n, k):
    """
    Calcule le coût de réplication d'un Short Call (Palmer 2001).
    Renvoie: (coût_initial, arbre_des_deltas)
    """
    S_tree = [[0.0 for j in range(i + 1)] for i in range(n + 1)]
    Delta_tree = [[0.0 for j in range(i + 1)] for i in range(n + 1)]
    B_tree = [[0.0 for j in range(i + 1)] for i in range(n + 1)]

    warning_printed = False  # Pour n'afficher le warning qu'une seule fois par appel

    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[i][j] = S0 * (u**j) * (d**(i - j))

    for j in range(n + 1):
        if S_tree[n][j] > K:
            Delta_tree[n][j] = -1.0
            B_tree[n][j] = K
        else:
            Delta_tree[n][j] = 0.0
            B_tree[n][j] = 0.0

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            S_node = S_tree[i][j]
            Delta_u, B_u = Delta_tree[i + 1][j + 1], B_tree[i + 1][j + 1]
            Delta_d, B_d = Delta_tree[i + 1][j], B_tree[i + 1][j]

            # Vérification de la condition d'unicité de Palmer (Théorème 2.1)
            if not warning_printed:
                if Delta_u < Delta_d and k * (u + d) / (u - d) >= 1:
                    print(f"Warning: Condition d'unicité non respectée pour k={k}, n={n} (Delta_u < Delta_d et k(u+d)/(u-d) >= 1)")
                    warning_printed = True
                elif Delta_u >= Delta_d and k >= 1:
                    print(f"Warning: Condition d'unicité non respectée pour k={k}, n={n} (Delta_u >= Delta_d et k >= 1)")
                    warning_printed = True

            C0 = (Delta_u * u - Delta_d * d) / (u - d) + (B_u - B_d) / (S_node * (u - d))
            
            def f(delta):
                return delta - C0 - (k / (u - d)) * (abs(Delta_u - delta) * u - abs(Delta_d - delta) * d)

            S_u_sign = 1.0 if f(Delta_u) >= 0 else -1.0
            S_d_sign = 1.0 if f(Delta_d) >= 0 else -1.0

            num = C0 + (k / (u - d)) * (S_u_sign * Delta_u * u - S_d_sign * Delta_d * d)
            den = 1.0 + (k / (u - d)) * (S_u_sign * u - S_d_sign * d)
            Delta_current = num / den

            B_current = (Delta_u * S_node * u + B_u + k * abs(Delta_u - Delta_current) * S_node * u - Delta_current * S_node * u) / R

            Delta_tree[i][j] = Delta_current
            B_tree[i][j] = B_current

    cost = - (Delta_tree[0][0] * S0 + B_tree[0][0])
    return cost, Delta_tree

# --- Variables initiales ---
S0 = 100.0
sigma = 0.20
r = math.log(1.1)
T = 1.0

ns = [6, 13, 52]
Ks = [80, 90, 100, 110, 120]
ks = [0, 0.00125, 0.005, 0.02]

results = []
for k in ks:
    for K in Ks:
        row = {'k': k, 'K': K}
        for n in ns:
            # Paramètre du papier
            dt = T / n
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
            R = math.exp(r * dt)
            
            cost, _ = vorst_algo(u, d, R, S0, K, n, k)
            row[f'n={n}'] = round(cost, 3)
        results.append(row)

df = pd.DataFrame(results)
#print(df.to_string(index=False))
#PS: on a le même résultat que sur la note de Boyle et Vorst par Palmer


def bs_call(S0, K, T, r_c, sigma):
    """
    Formule classique de Black-Scholes pour un Call européen.
    """
    if sigma <= 0: return max(S0 - K*math.exp(-r_c*T), 0)
    d1 = (math.log(S0 / K) + (r_c + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r_c * T) * norm.cdf(d2)



S0 = 100
sigma = 0.2
T = 1.0
r_eff = 0.1

r_c = math.log(1 + r_eff)
k_fixed = 0.001
K_fixed = 100
n_vals = np.arange(10, 3001, 100)

bv_prices_n = []
mod_bs_prices_n = []

for n in n_vals:
    # Prix discret
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    R = (1 + r_eff) ** dt 
    
    # On ne garde que la valeur 'cost'
    cost, _ = vorst_algo(u, d, R, S0, K_fixed, n, k_fixed)
    bv_prices_n.append(cost)
    
    # Volatilité modifiée de Boyle et Vorst
    mod_vol = sigma * math.sqrt(1 - (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T)))
    # Prix Black-Scholes avec volatilité modifiée
    mod_bs_p = bs_call(S0, K_fixed, T, r_c, mod_vol)
    mod_bs_prices_n.append(mod_bs_p)
    

plt.figure(figsize=(10,6))
plt.plot(n_vals, bv_prices_n, label='Modele Discret de Boyle-Vorst', marker='o', markersize=3)
plt.plot(n_vals, mod_bs_prices_n, label='Approximation de Black-Scholes Modifiee', linestyle='--')
plt.title("Convergence vers BS modifie en fonction du nombre de pas n")
plt.xlabel("Nombre de pas de discretisation (n)")
plt.ylabel("Prix de l'option Long Call")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence.png")


# --- 3. Convergence vers le modèle sans friction lorsque k tend vers 0 ---
n_fixed = 50
k_vals = np.linspace(0, 0.02, 50)
bv_prices_k = []
bs_standard = bs_call(S0, K_fixed, T, r_c, sigma)

for k_ in k_vals:
    # Il faut recalculer u, d, R pour n_fixed et utiliser vorst_algo
    dt = T / n_fixed
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = (1 + r_eff) ** dt 
    
    cost, _ = vorst_algo(u, d, R, S0, K_fixed, n_fixed, k_)
    bv_prices_k.append(cost)
    
plt.figure(figsize=(10,6))
plt.plot(k_vals, bv_prices_k, label=f'Modele Boyle-Vorst (n={n_fixed})', marker='x')
plt.axhline(bs_standard, color='r', linestyle='--', label='Black-Scholes (sans friction)')
plt.title("Convergence vers le modele sans friction lorsque k tend vers 0")
plt.xlabel("Cout de transaction (k)")
plt.ylabel("Prix de l'option Long Call")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_k.png")