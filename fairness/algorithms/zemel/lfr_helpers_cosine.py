# Based on code from https://github.com/zjelveh/learning-fair-representations
from numba.decorators import jit, njit
import numpy as np
from sklearn.metrics import confusion_matrix
import math
# import matplotlib.pyplot as plt


@jit
def distances(X, v, alpha, N, P, k):
    dists = np.zeros((N, k))
    for i in range(N):
        for j in range(k):
            for p in range(P):
                dists[i, j] += (X[i, p] - v[j, p]) * (X[i, p] - v[j, p]) * alpha[p]
            dists[i, j] = np.sqrt(dists[i, j])
    return dists


@njit(fastmath=True)
def compute_norm(matrix, alpha, rows, cols, print=True):
    alpha = np.sqrt(alpha)
    norm = np.zeros(rows)
    for i in range(rows):
        for p in range(cols):
            norm[i] += matrix[i, p] * matrix[i, p] * alpha[p]
        norm[i] = np.sqrt(norm[i]) if norm[i] > 0.0 else 1e-6
    return norm

@njit(fastmath=True)
def distances_cosine(X, X_norm, v, alpha, N, P, k):
    dists = np.zeros((N, k))
    v_norm = compute_norm(v, alpha, k, P)
    if np.sum(v_norm == 0.0) != 0: 
        print('v_norm contains zero: ', v_norm)
        print('alpha: ', alpha)
        print('prototypes: ', v)
    if np.sum(X_norm == 0.0) != 0:
        print("X_norm contains zero: ", X_norm)
    for i in range(N):
        for j in range(k):
            for p in range(P):
                dists[i, j] += X[i, p] * v[j, p] * alpha[p]
            if v_norm[j]:
                dists[i, j] = dists[i, j] / (X_norm[i] * v_norm[j])
            else:
                dists[i, j] = dists[i, j] / X_norm[i]
    return dists

@jit
def M_nk(dists, N, k):
    M_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk

@jit
def M_nk_cosine(dists, N, k):
    M_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk

@jit
def M_k(M_nk, N, k):
    M_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            M_k[j] += M_nk[i, j]
        M_k[j] /= N
    return M_k

@jit
def x_n_hat(X, M_nk, v, N, P, k):
    x_n_hat = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            L_x += (X[i, p] - x_n_hat[i, p]) * (X[i, p] - x_n_hat[i, p])
    return x_n_hat, L_x

@njit(fastmath=True)
def x_n_hat_cosine(X, M_nk, v, N, P, k):
    x_n_hat = np.zeros((N, P))
    x_hat_norm = np.zeros(N)
    x_norm = np.zeros(N)
    numerator = np.zeros(N)
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            x_hat_norm[i] += x_n_hat[i, p] * x_n_hat[i, p]
            x_norm[i] += X[i, p] * X[i, p]
            numerator[i] += X[i, p] * x_n_hat[i, p]
        if x_norm[i] and x_hat_norm[i]:
            cosine_cosine = numerator[i] / (np.sqrt(x_norm[i]) * np.sqrt(x_hat_norm[i]))
        else: 
            cosine_cosine = 0.0
        L_x += 1 - cosine_cosine
        if cosine_cosine > 1 or cosine_cosine <0:
            print("Something wrong with cosine: ", cosine_cosine)
    return x_n_hat, L_x


@jit
def yhat(M_nk, y, w, N, k):
    yhat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for j in range(k):
            yhat[i] += M_nk[i, j] * w[j]
        yhat[i] = 1e-6 if yhat[i] <= 0 else yhat[i]
        yhat[i] = 0.999 if yhat[i] >= 1 else yhat[i]
        L_y += -1 * y[i] * np.log(yhat[i]) - (1.0 - y[i]) * np.log(1.0 - yhat[i])
    return yhat, L_y


@jit
def LFR_optim_obj(params, data_sensitive, data_nonsensitive, y_sensitive,
                  y_nonsensitive, k=10, A_x = 0.01, A_y = 0.1, A_z = 0.5, results=0, print_inteval=250,
                  path_loss = None):
    
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape

    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.matrix(params[(2 * P) + k:]).reshape((k, P))

    # Compute norm for cosine distances prototype / input
    X_norm_sensitive = compute_norm(data_sensitive, alpha1, Ns, P)
    X_norm_nonsensitive = compute_norm(data_nonsensitive, alpha0, Nns, P)

    dists_sensitive = distances_cosine(data_sensitive, X_norm_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances_cosine(data_nonsensitive, X_norm_nonsensitive, v, alpha0, Nns, P, k)

    M_nk_sensitive = M_nk_cosine(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk_cosine(dists_nonsensitive, Nns, k)

    x_n_hat_sensitive, L_x1 = x_n_hat_cosine(data_sensitive, M_nk_sensitive, v, Ns, P, k)
    x_n_hat_nonsensitive, L_x2 = x_n_hat_cosine(data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k)
    L_x = (L_x1 + L_x2) / (Ns + Nns)

    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k)
    L_y = (L_y1 + L_y2) / (Ns + Nns) # Average cross entropy

    if A_z != 0.0: 
        M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
        M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)
        L_z = 0.0
        for j in range(k):
            L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])
        criterion = A_x * L_x + A_y * L_y + A_z * L_z
    else:
        criterion = A_y * L_y + A_x * L_x

    if LFR_optim_obj.iters % print_inteval == 0 or (LFR_optim_obj.iters) == 0:
        # print(LFR_optim_obj.iters, criterion)
        # print('\n')
        # print('dists_sensitive: ', dists_sensitive)
        # print("dists_nonsensitive: ", dists_nonsensitive)
        # print('alpha_plus: ', alpha1)
        # print('alpha_min: ', alpha0)
        # print('weights w: ', w)
        # print('prototypes: ', v)
        # print("M_nk_sensitive: ", M_nk_sensitive)
        # print("M_nk_nonsensitive: ", M_nk_nonsensitive)
        # print('yhat_sensitive: ', yhat_sensitive)
        # print(' yhat_nonsensitive: ', yhat_nonsensitive)
        # print('L_x_sens: ', L_x1)
        # print('L_x_nonsens: ', L_x2)
        # print('L_x: ', L_x)
        # print('L_y: ', L_y)
        # print('L_z: ', L_z)
        # print('\n')

        if math.isnan(L_y):
            # print('checking sensitive')
            # for i in range(Ns):
            #     if np.isnan(np.sum(M_nk_sensitive[i, :])):
            #         print('i: ', i)
            #         print('M_nk: ', M_nk_sensitive[i, :])
            #         print('dist: ', dists_sensitive[i, :])
            # print('checking nonsensitive')
            # for i in range(Nns):
            #     if np.isnan(np.sum(M_nk_nonsensitive[i, :])):
            #         print('i: ', i)
            #         print('M_nk: ', M_nk_nonsensitive[i, :])
            #         print('dist: ', dists_nonsensitive[i, :])
            print("Loss is NaN!!!")
            print('alpha1: ', alpha1)
            print('alpha0: ', alpha0)
            print('X_norm_nonsensitive = ', compute_norm(data_nonsensitive, alpha0, Nns, P))
            print('v_norm = ', compute_norm(v, alpha, k, P))
            exit()
        
        if path_loss is not None:
            f = open(path_loss, 'a+')
            f.write(str([L_x, L_y]) + "\n")

    LFR_optim_obj.iters += 1

    if results:
        return yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive
    else:
        return criterion
LFR_optim_obj.iters = 0
