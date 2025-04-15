import numpy as np
import random

random.seed(42)

def EM_algorithm(p, q, data, delta=1e-4, max_iter=100):
    n = len(data)
    for i in range(max_iter):
        # E-step
        p_old, q_old = p, q
        p_new = np.zeros(n)
        q_new = np.zeros(n)
        for j in range(n):
            p_new[j] = p**data[j].count('H') * (1-p)**data[j].count('T') / (p**data[j].count('H') * (1-p)**data[j].count('T') + q**data[j].count('H') * (1-q)**data[j].count('T'))
            q_new[j] = 1 - p_new[j]
        E_A_H = np.sum(p_new * np.array([data[j].count('H') for j in range(n)]))
        E_A_T = np.sum((p_new) * np.array([data[j].count('T') for j in range(n)]))
        E_B_H = np.sum(q_new * np.array([data[j].count('H') for j in range(n)]))
        E_B_T = np.sum((q_new) * np.array([data[j].count('T') for j in range(n)]))
        # M-step
        p = E_A_H / (E_A_H + E_A_T)
        q = E_B_H / (E_B_H + E_B_T)
        print('Iteration:', i+1)
        print('p =', p)
        print('q =', q)
        if np.abs(p-p_old) < delta and np.abs(q-q_old) < delta:
            break
    return p, q

data = [['H', 'T', 'T', 'T', 'H', 'H', 'T', 'H', 'T', 'H'],
        ['H', 'H', 'H', 'H', 'T', 'H', 'H', 'H', 'H', 'H'],
        ['H', 'T', 'H', 'H', 'H', 'H', 'H', 'T', 'H', 'H'],
        ['H', 'T', 'H', 'T', 'T', 'T', 'H', 'H', 'T', 'T'],
        ['T', 'H', 'H', 'H', 'T', 'H', 'H', 'H', 'T', 'H']]
p, q = EM_algorithm(0.6, 0.5, data)
print('result:')
print('\tp =', p)
print('\tq =', q)