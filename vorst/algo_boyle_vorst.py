import math
import pandas as pd

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
print(df.to_string(index=False))
#PS: on a le même résultat que sur la note de Boyle et Vorst par Palmer