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


def palmer_generic(u, d, R, S0, n, k, terminal_delta_fn, terminal_B_fn):
    """
    Algorithme générique de Palmer pour un dérivé quelconque.
    terminal_delta_fn(S) → Δ_T
    terminal_B_fn(S)     → B_T
    Renvoie le coût initial Δ₀·S₀ + B₀.
    """
    S_tree = [[0.0]*(i+1) for i in range(n+1)]
    Delta_tree = [[0.0]*(i+1) for i in range(n+1)]
    B_tree = [[0.0]*(i+1) for i in range(n+1)]

    for i in range(n+1):
        for j in range(i+1):
            S_tree[i][j] = S0 * (u**j) * (d**(i-j))

    for j in range(n+1):
        S_T = S_tree[n][j]
        Delta_tree[n][j] = terminal_delta_fn(S_T)
        B_tree[n][j] = terminal_B_fn(S_T)

    for i in range(n-1, -1, -1):
        for j in range(i+1):
            S_node = S_tree[i][j]
            Delta_u, B_u = Delta_tree[i+1][j+1], B_tree[i+1][j+1]
            Delta_d, B_d = Delta_tree[i+1][j], B_tree[i+1][j]

            C0 = (Delta_u * u - Delta_d * d) / (u - d) + (B_u - B_d) / (S_node * (u - d))

            def f(delta, _Du=Delta_u, _Dd=Delta_d, _C0=C0):
                return delta - _C0 - (k/(u-d)) * (abs(_Du - delta)*u - abs(_Dd - delta)*d)

            sign_u = 1.0 if f(Delta_u) >= 0 else -1.0
            sign_d = 1.0 if f(Delta_d) >= 0 else -1.0

            num = C0 + (k/(u-d)) * (sign_u * Delta_u * u - sign_d * Delta_d * d)
            den = 1.0 + (k/(u-d)) * (sign_u * u - sign_d * d)
            Delta_current = num / den if abs(den) > 1e-15 else C0

            B_current = (Delta_u * S_node * u + B_u
                         + k * abs(Delta_u - Delta_current) * S_node * u
                         - Delta_current * S_node * u) / R

            Delta_tree[i][j] = Delta_current
            B_tree[i][j] = B_current

    return Delta_tree[0][0] * S0 + B_tree[0][0]


# ═══════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES
# ═══════════════════════════════════════════════════════════════════════════

def bs_call(S0, K, T, r_c, sigma):
    if sigma <= 0: return max(S0 - K*math.exp(-r_c*T), 0)
    d1 = (math.log(S0 / K) + (r_c + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r_c * T) * norm.cdf(d2)

def bs_put(S0, K, T, r_c, sigma):
    if sigma <= 0: return max(K*math.exp(-r_c*T) - S0, 0)
    d1 = (math.log(S0/K) + (r_c + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r_c*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES COMMUNS
# ═══════════════════════════════════════════════════════════════════════════

S0 = 100.0
sigma = 0.20
r_eff = 0.1
r_c = math.log(1 + r_eff)
T = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  1. TABLE SHORT CALL (reproduction Palmer Table 2.1)
# ═══════════════════════════════════════════════════════════════════════════

ns = [6, 13, 52]
Ks = [80, 90, 100, 110, 120]
ks = [0, 0.00125, 0.005, 0.02]

results = []
for k in ks:
    for K in Ks:
        row = {'k': k, 'K': K}
        for n in ns:
            dt = T / n
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
            R = math.exp(r_c * dt)
            cost, _ = vorst_algo(u, d, R, S0, K, n, k)
            row[f'n={n}'] = round(cost, 3)
        results.append(row)

df = pd.DataFrame(results)
print("Table Short Call (Palmer 2001)")
print(df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
#  2. CONVERGENCE SHORT CALL : discret vs BS vol. modifiée
# ═══════════════════════════════════════════════════════════════════════════

K_fixed = 100
k_fixed = 0.001
n_vals = np.arange(10, 3001, 100)

bv_prices_n = []
mod_bs_prices_n = []

for n in n_vals:
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    R = (1 + r_eff) ** dt
    cost, _ = vorst_algo(u, d, R, S0, K_fixed, n, k_fixed)
    bv_prices_n.append(cost)
    mod_vol = sigma * math.sqrt(1 - (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T)))
    mod_bs_p = bs_call(S0, K_fixed, T, r_c, mod_vol)
    mod_bs_prices_n.append(mod_bs_p)

plt.figure(figsize=(10,6))
plt.plot(n_vals, bv_prices_n, label='Modele Discret de Boyle-Vorst', marker='o', markersize=3)
plt.plot(n_vals, mod_bs_prices_n, label='Approximation de Black-Scholes Modifiee', linestyle='--')
plt.title("Short Call : convergence vers BS modifie en fonction de n")
plt.xlabel("Nombre de pas de discretisation (n)")
plt.ylabel("Prix de l'option")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  3. CONVERGENCE k → 0 (Short Call)
# ═══════════════════════════════════════════════════════════════════════════

n_fixed = 50
k_vals = np.linspace(0, 0.02, 50)
bv_prices_k = []
bs_standard = bs_call(S0, K_fixed, T, r_c, sigma)

for k_ in k_vals:
    dt = T / n_fixed
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = (1 + r_eff) ** dt
    cost, _ = vorst_algo(u, d, R, S0, K_fixed, n_fixed, k_)
    bv_prices_k.append(cost)

plt.figure(figsize=(10,6))
plt.plot(k_vals, bv_prices_k, label=f'Modele Boyle-Vorst (n={n_fixed})', marker='x')
plt.axhline(bs_standard, color='r', linestyle='--', label='Black-Scholes (sans friction)')
plt.title("Short Call : convergence vers le modele sans friction (k -> 0)")
plt.xlabel("Cout de transaction (k)")
plt.ylabel("Prix de l'option")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  4. BULL CALL SPREAD  (Long Call K1 + Short Call K2,  K1 < K2)
# ═══════════════════════════════════════════════════════════════════════════

K1, K2 = 90, 110

def bull_delta(S):
    if S > K2:   return 0.0
    elif S > K1: return 1.0
    else:        return 0.0

def bull_B(S):
    if S > K2:   return K2 - K1
    elif S > K1: return -K1
    else:        return 0.0

bs_bull = bs_call(S0, K1, T, r_c, sigma) - bs_call(S0, K2, T, r_c, sigma)

k_fixed = 0.001
n_vals_conv = np.arange(10, 1001, 20)
n_fix = 100
k_range = np.linspace(0, 0.015, 60)

bull_prices = []
bull_mod_bs = []
for n in n_vals_conv:
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = math.exp(r_c * dt)
    bull_prices.append(palmer_generic(u, d, R, S0, n, k_fixed, bull_delta, bull_B))
    arg = 1 + (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    mod_vol = sigma * math.sqrt(arg)
    bull_mod_bs.append(bs_call(S0, K1, T, r_c, mod_vol) - bs_call(S0, K2, T, r_c, mod_vol))

plt.figure(figsize=(10,6))
plt.plot(n_vals_conv, bull_prices, 'b-', label='Discret (Palmer)', linewidth=1.5)
plt.plot(n_vals_conv, bull_mod_bs, 'r--', label='BS vol. modifiee (BV)')
plt.axhline(bs_bull, color='g', linestyle=':', label=f'BS standard = {bs_bull:.3f}')
plt.title(f"Bull Call Spread ({K1}/{K2}) — convergence en n  (k={k_fixed})")
plt.xlabel("n"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

bull_k_prices = []
for kk in k_range:
    dt = T / n_fix; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    bull_k_prices.append(palmer_generic(u, d, R, S0, n_fix, kk, bull_delta, bull_B))

plt.figure(figsize=(10,6))
plt.plot(k_range, bull_k_prices, 'b-', label=f'Bull Spread (n={n_fix})')
plt.axhline(bs_bull, color='r', linestyle='--', label=f'BS = {bs_bull:.3f}')
plt.title(f"Bull Call Spread ({K1}/{K2}) — convergence k -> 0")
plt.xlabel("k"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  5. STRADDLE  (Long Call K + Long Put K)
# ═══════════════════════════════════════════════════════════════════════════

K_str = 100

def straddle_delta(S):
    return 1.0 if S >= K_str else -1.0

def straddle_B(S):
    return -K_str if S >= K_str else K_str

bs_straddle = bs_call(S0, K_str, T, r_c, sigma) + bs_put(S0, K_str, T, r_c, sigma)

straddle_prices = []
straddle_mod_bs = []
for n in n_vals_conv:
    dt = T / n; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    straddle_prices.append(palmer_generic(u, d, R, S0, n, k_fixed, straddle_delta, straddle_B))
    arg = 1 + (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    mod_vol = sigma * math.sqrt(arg)
    straddle_mod_bs.append(bs_call(S0, K_str, T, r_c, mod_vol) + bs_put(S0, K_str, T, r_c, mod_vol))

plt.figure(figsize=(10,6))
plt.plot(n_vals_conv, straddle_prices, 'b-', label='Discret (Palmer)', linewidth=1.5)
plt.plot(n_vals_conv, straddle_mod_bs, 'r--', label='BS vol. modifiee (BV)')
plt.axhline(bs_straddle, color='g', linestyle=':', label=f'BS standard = {bs_straddle:.3f}')
plt.title(f"Straddle (K={K_str}) — convergence en n  (k={k_fixed})")
plt.xlabel("n"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

straddle_k_prices = []
for kk in k_range:
    dt = T / n_fix; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    straddle_k_prices.append(palmer_generic(u, d, R, S0, n_fix, kk, straddle_delta, straddle_B))

plt.figure(figsize=(10,6))
plt.plot(k_range, straddle_k_prices, 'b-', label=f'Straddle (n={n_fix})')
plt.axhline(bs_straddle, color='r', linestyle='--', label=f'BS = {bs_straddle:.3f}')
plt.title(f"Straddle (K={K_str}) — convergence k -> 0")
plt.xlabel("k"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  6. BUTTERFLY SPREAD  (Long Call K−a, Short 2×Call K, Long Call K+a)
# ═══════════════════════════════════════════════════════════════════════════

K_bf = 100
a_bf = 10

def butterfly_delta(S):
    if S <= K_bf - a_bf:   return 0.0
    elif S <= K_bf:        return 1.0
    elif S <= K_bf + a_bf: return -1.0
    else:                  return 0.0

def butterfly_B(S):
    if S <= K_bf - a_bf:   return 0.0
    elif S <= K_bf:        return -(K_bf - a_bf)
    elif S <= K_bf + a_bf: return (K_bf + a_bf)
    else:                  return 0.0

bs_butterfly = (bs_call(S0, K_bf - a_bf, T, r_c, sigma)
                - 2 * bs_call(S0, K_bf, T, r_c, sigma)
                + bs_call(S0, K_bf + a_bf, T, r_c, sigma))

butterfly_prices = []
butterfly_mod_bs = []
for n in n_vals_conv:
    dt = T / n; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    butterfly_prices.append(palmer_generic(u, d, R, S0, n, k_fixed, butterfly_delta, butterfly_B))
    arg = 1 + (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    mod_vol = sigma * math.sqrt(arg)
    butterfly_mod_bs.append(bs_call(S0, K_bf-a_bf, T, r_c, mod_vol) - 2*bs_call(S0, K_bf, T, r_c, mod_vol) + bs_call(S0, K_bf+a_bf, T, r_c, mod_vol))

plt.figure(figsize=(10,6))
plt.plot(n_vals_conv, butterfly_prices, 'b-', label='Discret (Palmer)', linewidth=1.5)
plt.plot(n_vals_conv, butterfly_mod_bs, 'r--', label='BS vol. modifiee (BV)')
plt.axhline(bs_butterfly, color='g', linestyle=':', label=f'BS standard = {bs_butterfly:.3f}')
plt.title(f"Butterfly Spread ({K_bf-a_bf}/{K_bf}/{K_bf+a_bf}) — convergence en n  (k={k_fixed})")
plt.xlabel("n"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

butterfly_k_prices = []
for kk in k_range:
    dt = T / n_fix; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    butterfly_k_prices.append(palmer_generic(u, d, R, S0, n_fix, kk, butterfly_delta, butterfly_B))

plt.figure(figsize=(10,6))
plt.plot(k_range, butterfly_k_prices, 'b-', label=f'Butterfly (n={n_fix})')
plt.axhline(bs_butterfly, color='r', linestyle='--', label=f'BS = {bs_butterfly:.3f}')
plt.title(f"Butterfly Spread ({K_bf-a_bf}/{K_bf}/{K_bf+a_bf}) — convergence k -> 0")
plt.xlabel("k"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  7. BEAR PUT SPREAD  (Long Put K2 + Short Put K1,  K1 < K2)
# ═══════════════════════════════════════════════════════════════════════════

K1_bp, K2_bp = 90, 110

def bear_put_delta(S):
    if S < K1_bp:    return 0.0
    elif S < K2_bp:  return -1.0
    else:            return 0.0

def bear_put_B(S):
    if S < K1_bp:    return K2_bp - K1_bp
    elif S < K2_bp:  return K2_bp
    else:            return 0.0

bs_bear_put = bs_put(S0, K2_bp, T, r_c, sigma) - bs_put(S0, K1_bp, T, r_c, sigma)

bear_prices = []
bear_mod_bs = []
for n in n_vals_conv:
    dt = T / n; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    bear_prices.append(palmer_generic(u, d, R, S0, n, k_fixed, bear_put_delta, bear_put_B))
    arg = 1 + (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    mod_vol = sigma * math.sqrt(arg)
    bear_mod_bs.append(bs_put(S0, K2_bp, T, r_c, mod_vol) - bs_put(S0, K1_bp, T, r_c, mod_vol))

plt.figure(figsize=(10,6))
plt.plot(n_vals_conv, bear_prices, 'b-', label='Discret (Palmer)', linewidth=1.5)
plt.plot(n_vals_conv, bear_mod_bs, 'r--', label='BS vol. modifiee (BV)')
plt.axhline(bs_bear_put, color='g', linestyle=':', label=f'BS standard = {bs_bear_put:.3f}')
plt.title(f"Bear Put Spread ({K1_bp}/{K2_bp}) — convergence en n  (k={k_fixed})")
plt.xlabel("n"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

bear_k_prices = []
for kk in k_range:
    dt = T / n_fix; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    bear_k_prices.append(palmer_generic(u, d, R, S0, n_fix, kk, bear_put_delta, bear_put_B))

plt.figure(figsize=(10,6))
plt.plot(k_range, bear_k_prices, 'b-', label=f'Bear Put Spread (n={n_fix})')
plt.axhline(bs_bear_put, color='r', linestyle='--', label=f'BS = {bs_bear_put:.3f}')
plt.title(f"Bear Put Spread ({K1_bp}/{K2_bp}) — convergence k -> 0")
plt.xlabel("k"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  8. STRANGLE  (Long Put K1 + Long Call K2,  K1 < K2)
# ═══════════════════════════════════════════════════════════════════════════

K1_sg, K2_sg = 90, 110

def strangle_delta(S):
    if S < K1_sg:    return -1.0
    elif S <= K2_sg: return 0.0
    else:            return 1.0

def strangle_B(S):
    if S < K1_sg:    return K1_sg
    elif S <= K2_sg: return 0.0
    else:            return -K2_sg

bs_strangle = bs_put(S0, K1_sg, T, r_c, sigma) + bs_call(S0, K2_sg, T, r_c, sigma)

strangle_prices = []
strangle_mod_bs = []
for n in n_vals_conv:
    dt = T / n; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    strangle_prices.append(palmer_generic(u, d, R, S0, n, k_fixed, strangle_delta, strangle_B))
    arg = 1 + (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    mod_vol = sigma * math.sqrt(arg)
    strangle_mod_bs.append(bs_put(S0, K1_sg, T, r_c, mod_vol) + bs_call(S0, K2_sg, T, r_c, mod_vol))

plt.figure(figsize=(10,6))
plt.plot(n_vals_conv, strangle_prices, 'b-', label='Discret (Palmer)', linewidth=1.5)
plt.plot(n_vals_conv, strangle_mod_bs, 'r--', label='BS vol. modifiee (BV)')
plt.axhline(bs_strangle, color='g', linestyle=':', label=f'BS standard = {bs_strangle:.3f}')
plt.title(f"Strangle ({K1_sg}/{K2_sg}) — convergence en n  (k={k_fixed})")
plt.xlabel("n"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

strangle_k_prices = []
for kk in k_range:
    dt = T / n_fix; u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u; R = math.exp(r_c * dt)
    strangle_k_prices.append(palmer_generic(u, d, R, S0, n_fix, kk, strangle_delta, strangle_B))

plt.figure(figsize=(10,6))
plt.plot(k_range, strangle_k_prices, 'b-', label=f'Strangle (n={n_fix})')
plt.axhline(bs_strangle, color='r', linestyle='--', label=f'BS = {bs_strangle:.3f}')
plt.title(f"Strangle ({K1_sg}/{K2_sg}) — convergence k -> 0")
plt.xlabel("k"); plt.ylabel("Prix"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  9. RÉSUMÉ : convergence en n, toutes options sur une figure
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

all_options = [
    ("Bull Spread 90/110",     bull_prices,      bs_bull),
    ("Straddle K=100",         straddle_prices,  bs_straddle),
    ("Butterfly 90/100/110",   butterfly_prices,  bs_butterfly),
    ("Bear Put Spread 90/110", bear_prices,       bs_bear_put),
    ("Strangle 90/110",        strangle_prices,   bs_strangle),
]

for ax, (name, prices, bs_ref) in zip(axes, all_options):
    ax.plot(n_vals_conv, prices, 'b-', linewidth=1.5, label='Discret (Palmer)')
    ax.axhline(bs_ref, color='r', linestyle='--', label=f'BS = {bs_ref:.3f}')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel("n")
    ax.set_ylabel("Prix")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_visible(False)
fig.suptitle(f"Convergence en n — options exotiques avec couts de transaction\n"
             f"(S0={S0}, sigma={sigma}, r={r_eff:.0%}, T={T}, k={k_fixed})",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Résumé convergence k → 0
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

all_k_options = [
    ("Bull Spread 90/110",     bull_k_prices,      bs_bull),
    ("Straddle K=100",         straddle_k_prices,  bs_straddle),
    ("Butterfly 90/100/110",   butterfly_k_prices,  bs_butterfly),
    ("Bear Put Spread 90/110", bear_k_prices,       bs_bear_put),
    ("Strangle 90/110",        strangle_k_prices,   bs_strangle),
]

for ax, (name, prices, bs_ref) in zip(axes, all_k_options):
    ax.plot(k_range, prices, 'b-', linewidth=1.5, label=f'{name} (n={n_fix})')
    ax.axhline(bs_ref, color='r', linestyle='--', label=f'BS = {bs_ref:.3f}')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel("k")
    ax.set_ylabel("Prix")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_visible(False)
fig.suptitle(f"Convergence k -> 0 — options exotiques\n"
             f"(S0={S0}, sigma={sigma}, r={r_eff:.0%}, T={T}, n={n_fix})",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  10. SHORT PUT  (obligation d'acheter si S < K)
#      Terminal: si S < K → Δ=1, B=−K ; sinon Δ=0, B=0
#      Le coût de réplication du short put est Δ₀·S₀ + B₀.
#      La borne inférieure du prix du put = −(coût de réplication du short put)
#
#      Le Théorème 1 de Boyle-Vorst (monotonie Δ₂ ≤ Δ ≤ Δ₁) est vérifié
#      car le delta terminal est croissant en S (0 pour S ≥ K, 1 pour S < K
#      → quand on monte dans l'arbre, Δ_u ≤ Δ_d n'arrive jamais en inversé
#      pour un short put avec delivery). L'approximation BS à vol. modifiée
#      avec σ²(1 − 2k√n / σ√T) est donc applicable (comme pour le short call).
# ═══════════════════════════════════════════════════════════════════════════

K_sp = 100

def short_put_delta(S):
    """Short Put delivery: si S < K, l'émetteur doit acheter l'action → Δ=+1."""
    return 1.0 if S < K_sp else 0.0

def short_put_B(S):
    """Short Put delivery: si S < K, l'émetteur paie K → B=−K."""
    return -K_sp if S < K_sp else 0.0

bs_put_ref = bs_put(S0, K_sp, T, r_c, sigma)

# --- 10a. Convergence en n : discret vs BS vol. modifiée ---
print("\n--- Short Put : convergence en n ---")

n_vals_sp = np.arange(10, 2001, 50)
sp_discrete = []
sp_mod_bs = []

for n in n_vals_sp:
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = math.exp(r_c * dt)

    # Coût de réplication du short put
    cost = palmer_generic(u, d, R, S0, n, k_fixed, short_put_delta, short_put_B)
    # Borne inférieure du put = −cost
    sp_discrete.append(-cost)

    # Vol. modifiée short : σ²(1 − 2k√n / σ√T)
    arg = 1 - (2 * k_fixed * math.sqrt(n)) / (sigma * math.sqrt(T))
    if arg > 0:
        mod_vol = sigma * math.sqrt(arg)
        sp_mod_bs.append(bs_put(S0, K_sp, T, r_c, mod_vol))
    else:
        sp_mod_bs.append(float('nan'))

plt.figure(figsize=(10, 6))
plt.plot(n_vals_sp, sp_discrete, 'b-', linewidth=1.5,
         label='Modèle discret (Palmer) — borne inf.')
plt.plot(n_vals_sp, sp_mod_bs, 'r--', linewidth=1.2,
         label='BS vol. modifiée (BV) — short')
plt.axhline(bs_put_ref, color='g', linestyle=':', linewidth=1,
            label=f'BS standard = {bs_put_ref:.3f}')
plt.title(f"Short Put (K={K_sp}) — convergence en n  (k={k_fixed})")
plt.xlabel("Nombre de pas n")
plt.ylabel("Prix (borne inférieure du put)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 10b. Convergence en k → 0 ---
print("--- Short Put : convergence en k → 0 ---")

sp_k_prices = []
for kk in k_range:
    dt = T / n_fix
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = math.exp(r_c * dt)
    cost = palmer_generic(u, d, R, S0, n_fix, kk, short_put_delta, short_put_B)
    sp_k_prices.append(-cost)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sp_k_prices, 'b-', linewidth=1.5,
         label=f'Short Put (n={n_fix}) — borne inf.')
plt.axhline(bs_put_ref, color='r', linestyle='--',
            label=f'BS = {bs_put_ref:.3f}')
plt.title(f"Short Put (K={K_sp}) — convergence k → 0  (n={n_fix})")
plt.xlabel("k (coût de transaction)")
plt.ylabel("Prix (borne inférieure du put)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 10c. Bornes du Put : Long Put (borne sup) + Short Put (borne inf) ---
print("--- Put : bornes sup et inf ensemble ---")

n_vals_bounds = np.arange(10, 501, 10)

def long_put_delta(S):
    return -1.0 if S < K_sp else 0.0

def long_put_B(S):
    return K_sp if S < K_sp else 0.0

put_upper = []
put_lower = []

for n in n_vals_bounds:
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = math.exp(r_c * dt)

    # Borne sup (long put) = coût de réplication
    cost_long = palmer_generic(u, d, R, S0, n, k_fixed, long_put_delta, long_put_B)
    put_upper.append(cost_long)

    # Borne inf (short put) = −coût de réplication
    cost_short = palmer_generic(u, d, R, S0, n, k_fixed, short_put_delta, short_put_B)
    put_lower.append(-cost_short)

plt.figure(figsize=(10, 6))
plt.fill_between(n_vals_bounds, put_lower, put_upper, alpha=0.15, color='steelblue',
                 label='Intervalle [borne inf, borne sup]')
plt.plot(n_vals_bounds, put_upper, 'b-', linewidth=1.2, label='Borne sup. (Long Put)')
plt.plot(n_vals_bounds, put_lower, 'r-', linewidth=1.2, label='Borne inf. (Short Put)')
plt.axhline(bs_put_ref, color='g', linestyle=':', linewidth=1.5,
            label=f'BS = {bs_put_ref:.3f}')
plt.title(f"Put (K={K_sp}) — bornes de prix  (k={k_fixed})")
plt.xlabel("n")
plt.ylabel("Prix du Put")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()