"""
Author: Zeyu Wang
Date: October 2026
Description: "analytical ptychogrphay"
"""

import numpy as np
import time
import torch

from .utils import print_and_log, log_to_file, get_subplot_grid, RGB_Complex_Plot, calculate_wavelength, Chi_defocus





# @torch.no_grad()
def nndsvd_init(X, rank, seed=0, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    U, S, V = torch.pca_lowrank(
        X, q=rank, center=False, niter=2
    )

    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]

    W = torch.zeros((X.shape[0], rank), dtype=torch.float32)
    H = torch.zeros((rank, X.shape[1]), dtype=torch.float32)

    # First component: absolute leading singular vectors.
    W[:, 0] = torch.sqrt(S[0]) * U[:, 0].abs()
    H[0, :] = torch.sqrt(S[0]) * V[:, 0].abs()

    # Remaining components: choose the positive or negative
    # parts according to the larger product of their norms.
    for k in range(1, rank):
        u = U[:, k]
        v = V[:, k]

        u_pos = torch.clamp(u, min=0)
        u_neg = torch.clamp(-u, min=0)
        v_pos = torch.clamp(v, min=0)
        v_neg = torch.clamp(-v, min=0)

        n_up = torch.linalg.vector_norm(u_pos)
        n_un = torch.linalg.vector_norm(u_neg)
        n_vp = torch.linalg.vector_norm(v_pos)
        n_vn = torch.linalg.vector_norm(v_neg)

        score_pos = n_up * n_vp
        score_neg = n_un * n_vn

        if score_pos >= score_neg:
            u_sel, v_sel = u_pos, v_pos
            norm_u, norm_v = n_up, n_vp
        else:
            u_sel, v_sel = u_neg, v_neg
            norm_u, norm_v = n_un, n_vn

        if norm_u > 1e-12 and norm_v > 1e-12:
            scale = torch.sqrt(S[k] * norm_u * norm_v)
            W[:, k] = scale * u_sel / norm_u
            H[k, :] = scale * v_sel / norm_v

    # Small positive initialization for zero entries.
    avg = X.mean().item()
    eps = max(avg * 1e-3, 1e-6)
    W[W == 0] = eps
    H[H == 0] = eps

    return W.to(device), H.to(device)




# @torch.no_grad()
def fast_hals_NMF(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, device='None'):

    if device is None:
        device = "cpu"

    print_and_log('')
    print_and_log('##################### NMF Parameters ######################')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'NMF Components Number: {n_components}')
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'NMF Convergence Tolerance: {tol}')
    print_and_log(f'NMF Block Number: {n_block}')
    print_and_log(f'NMF alpha_W: {alpha_W}')
    print_and_log(f'NMF alpha_H: {alpha_H}')
    print_and_log(f'NMF L1 Ratio: {l1_ratio}')

    sy, sx, dy, dx = data_4D.shape
    X_np = np.ascontiguousarray(data_4D.reshape(-1, np.prod((dy, dx))), dtype=np.float32)

    # Nonnegative input is required by NMF.
    X_np = np.maximum(X_np, 0)
    X_cpu = torch.from_numpy(X_np)

    N, M = X_cpu.shape
    block_size = (N + n_block - 1) // n_block
    W, H = nndsvd_init(X_cpu, n_components, device=device)
    eps = 1e-12

    reg_W_l1 = alpha_W * M * l1_ratio
    reg_W_l2 = alpha_W * M * (1 - l1_ratio)
    reg_H_l1 = alpha_H * N * l1_ratio
    reg_H_l2 = alpha_H * N * (1 - l1_ratio)

    # Precompute ||X||_F^2 only once.
    norm_X_sq = torch.zeros((), dtype=torch.float64)
    for start in range(0, N, block_size):
        Xb = X_cpu[start:start + block_size]
        norm_X_sq += Xb.double().square().sum()
    norm_X_sq = norm_X_sq.item()

    err_list = []
    previous_err = None


    start_time = time.perf_counter()
    print_and_log('')
    print_and_log(f'####################### NMF Process ######################')
    print("\rNMF progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0, "*" * 0, "." * 10, 0, 0, 0), end="")

    for i in range(iter_max):

        time_i0 = time.perf_counter()

        # ----------------------------------------------------
        # Update H
        # Q = W.T @ X, A = W.T @ W
        # ----------------------------------------------------
        Q = torch.zeros((n_components, M), dtype=torch.float32, device=device)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = X_cpu[start:end].to(device)
            Q += W[start:end].T @ Xb

        A = W.T @ W

        for k in range(n_components):
            denom = (A[k, k] + reg_H_l2).clamp_min(eps)
            H[k] = torch.clamp(H[k] + (Q[k] - A[k] @ H - reg_H_l1) / denom, min=0)

        # ----------------------------------------------------
        # Update W
        # P = X @ H.T, B = H @ H.T
        # ----------------------------------------------------
        P = torch.empty((N, n_components), dtype=torch.float32, device=device)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = X_cpu[start:end].to(device)
            P[start:end] = Xb @ H.T

        B = H @ H.T

        for k in range(n_components):
            denom = (B[k, k] + reg_W_l2).clamp_min(eps)
            W[:, k] = torch.clamp(W[:, k] + (P[:, k] - W @ B[:, k] - reg_W_l1) / denom, min=0)

        # ----------------------------------------------------
        # Fast error calculation: reuse P = X @ H.T
        # ||X - WH||^2 = ||X||^2 - 2 <P,W> + <W.T W,H H.T>
        # ----------------------------------------------------
        A_final = W.T @ W

        cross_term = torch.sum(P.double() * W.double())
        model_norm_sq = torch.sum(A_final.double() * B.double())
        residual_sq = max(norm_X_sq - 2.0 * cross_term.item() + model_norm_sq.item(), 0.0)

        err = np.sqrt(residual_sq / max(norm_X_sq, eps))
        err_list.append(err)

        # Convergence check
        if previous_err is not None:
            relative_change = abs(previous_err - err) / max(previous_err, eps)
            if relative_change < tol:
                print('')
                print('error relative change < tol, converged')
                break

        previous_err = err

        time_is = time.perf_counter() - time_i0
        speed = 1 / time_is
        process = (i + 1) / iter_max * 100
        aa = "*" * int(process / 10)
        bb = "." * (10 - int(process / 10))
        dur = int(time.perf_counter() - start_time)
        time_remain = int((iter_max - i - 1) * time_is)
        dur_h = dur // 3600
        dur_m = (dur - dur_h * 3600) // 60
        dur_s = dur - dur_h * 3600 - dur_m * 60
        time_remain_h = time_remain // 3600
        time_remain_m = (time_remain - time_remain_h * 3600) // 60
        time_remain_s = time_remain - time_remain_h * 3600 - time_remain_m * 60

        print("\rNMF progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s), end="")
    print('')
    log_to_file("NMF progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s))
    
    gpu_time = time.perf_counter() - start_time
    print_and_log(f'NMF process finished in {gpu_time} s')

    W = np.moveaxis(np.array(W.cpu()).reshape(sy, sx, n_components), -1, 0)
    H = np.array(H.cpu()).reshape(n_components, dy, dx)

    return W, H, err_list



# Perform PCA using randomized SVD.
def PCA(data_4D, n_components, n_block=1, device='None', n_oversamples=10, n_power_iter=2, random_state=0):

    if device is None or device == 'None':
        device = 'cpu'

    print_and_log('')
    print_and_log('##################### PCA Parameters ######################')
    print_and_log(f'Device Using: {device}')
    print_and_log(f'PCA Components Number: {n_components}')
    print_and_log(f'PCA Block Number: {n_block}')
    print_and_log(f'PCA Oversamples: {n_oversamples}')
    print_and_log(f'PCA Power Iterations: {n_power_iter}')

    start_time = time.perf_counter()

    # --------------------------------------------------
    # 0. Prepare data
    # --------------------------------------------------
    sy, sx, dy, dx = data_4D.shape
    X_np = np.ascontiguousarray(data_4D.reshape(-1, dy * dx), dtype=np.float32)

    N, M = X_np.shape
    K = n_components

    if not isinstance(n_block, int) or n_block < 1:
        raise ValueError('n_block must be a positive integer.')

    if not isinstance(K, int) or not 1 <= K <= min(N - 1, M):
        raise ValueError(f'n_components must be between 1 and {min(N - 1, M)}.')

    if not isinstance(n_oversamples, int) or n_oversamples < 0:
        raise ValueError('n_oversamples must be a non-negative integer.')

    if not isinstance(n_power_iter, int) or n_power_iter < 0:
        raise ValueError('n_power_iter must be a non-negative integer.')

    dev = torch.device(device)
    block_size = min((N + n_block - 1) // n_block, 256)
    L = min(K + n_oversamples, N - 1, M)

    torch.manual_seed(random_state)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(random_state)

    # --------------------------------------------------
    # 1. Compute mean in blocks
    # --------------------------------------------------
    mean = torch.zeros(M, dtype=torch.float64, device=dev)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float64, device=dev)
        mean += Xb.sum(dim=0)

    mean /= N
    mean32 = mean.to(torch.float32)

    # --------------------------------------------------
    # 2. Randomized range finder
    # Y = Xc @ Omega
    # --------------------------------------------------
    Omega = torch.randn(M, L, dtype=torch.float32, device=dev)
    Y = torch.empty((N, L), dtype=torch.float32, device=dev)
    total_variance = 0.0

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
        Xc = Xb - mean32
        Y[start:end] = Xc @ Omega
        total_variance += Xc.double().square().sum().item()

    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # --------------------------------------------------
    # 3. Power iterations
    # Improve approximation of the leading PCA subspace
    # --------------------------------------------------
    for _ in range(n_power_iter):
        Z = torch.zeros((M, L), dtype=torch.float32, device=dev)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
            Xc = Xb - mean32
            Z += Xc.T @ Q[start:end]

        Z, _ = torch.linalg.qr(Z, mode='reduced')

        Y = torch.empty((N, L), dtype=torch.float32, device=dev)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
            Xc = Xb - mean32
            Y[start:end] = Xc @ Z

        Q, _ = torch.linalg.qr(Y, mode='reduced')

    # --------------------------------------------------
    # 4. SVD of compressed matrix
    # B = Q.T @ Xc
    # --------------------------------------------------
    B = torch.zeros((L, M), dtype=torch.float32, device=dev)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
        Xc = Xb - mean32
        B += Q[start:end].T @ Xc

    _, S, Vh = torch.linalg.svd(B, full_matrices=False)

    S = S[:K]
    H = Vh[:K].contiguous()

    # Eigenvalues of covariance matrix C = Xc.T @ Xc / N
    eigvals = S.double().square() / N
    Lambda = torch.diag(eigvals)

    explained_ratio = S.double().square().sum().item() / max(total_variance, 1e-12)

    # --------------------------------------------------
    # 5. Compute spatial scores in blocks
    # W = Xc @ H.T
    # --------------------------------------------------
    W = torch.empty((N, K), dtype=torch.float32, device=dev)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
        Xc = Xb - mean32
        W[start:end] = Xc @ H.T

    # --------------------------------------------------
    # 6. Reshape and return
    # --------------------------------------------------
    W = W.T.reshape(K, sy, sx).cpu().numpy()
    H = H.reshape(K, dy, dx).cpu().numpy()
    Lambda = Lambda.cpu().numpy()

    print_and_log(f'Explained Variance Ratio: {explained_ratio:.6f}')
    print_and_log(f'PCA process finished in {time.perf_counter() - start_time:.2f} s')

    return W, H, Lambda




# Perform FastICA with randomized PCA whitening.
def fast_ICA(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, device='None', n_oversamples=10, n_power_iter=2, random_state=0):

    if device is None or device == 'None':
        device = 'cpu'

    print_and_log('')
    print_and_log('###################### ICA Parameters ######################')
    print_and_log(f'Device Using: {device}')
    print_and_log(f'ICA Components Number: {n_components}')
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'ICA Convergence Tolerance: {tol}')
    print_and_log(f'ICA Block Number: {n_block}')
    print_and_log(f'PCA Oversamples: {n_oversamples}')
    print_and_log(f'PCA Power Iterations: {n_power_iter}')

    start_time = time.perf_counter()

    # --------------------------------------------------
    # 0. Prepare data
    # --------------------------------------------------
    sy, sx, dy, dx = data_4D.shape
    X_np = np.ascontiguousarray(data_4D.reshape(-1, dy * dx), dtype=np.float32)

    N, M = X_np.shape
    K = n_components

    if not isinstance(K, int) or not 1 <= K <= min(N - 1, M):
        raise ValueError(f'n_components must be between 1 and {min(N - 1, M)}.')

    if not isinstance(n_block, int) or n_block < 1:
        raise ValueError('n_block must be a positive integer.')

    if not isinstance(n_oversamples, int) or n_oversamples < 0:
        raise ValueError('n_oversamples must be a non-negative integer.')

    if not isinstance(n_power_iter, int) or n_power_iter < 0:
        raise ValueError('n_power_iter must be a non-negative integer.')

    if iter_max < 1 or tol <= 0:
        raise ValueError('iter_max must be positive and tol must be positive.')

    dev = torch.device(device)
    block_size = min((N + n_block - 1) // n_block, 256)
    L = min(K + n_oversamples, N - 1, M)

    torch.manual_seed(random_state)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(random_state)

    # --------------------------------------------------
    # 1. Mean
    # --------------------------------------------------
    mean = torch.zeros(M, dtype=torch.float64, device=dev)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float64, device=dev)
        mean += Xb.sum(dim=0)

    mean /= N
    mean32 = mean.to(torch.float32)

    # --------------------------------------------------
    # 2. Randomized truncated SVD for PCA whitening
    # --------------------------------------------------
    # Compute only the leading L principal components.
    # Avoid constructing the full M x M covariance matrix.

    Omega = torch.randn(M, L, dtype=torch.float32, device=dev)
    Y = torch.empty((N, L), dtype=torch.float32, device=dev)

    norm_X_sq = 0.0

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
        Xc = Xb - mean32
        Y[start:end] = Xc @ Omega
        norm_X_sq += Xc.double().square().sum().item()

    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # Power iterations improve the approximation of the leading subspace.
    for _ in range(n_power_iter):
        Z_range = torch.zeros((M, L), dtype=torch.float32, device=dev)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
            Xc = Xb - mean32
            Z_range += Xc.T @ Q[start:end]

        Z_range, _ = torch.linalg.qr(Z_range, mode='reduced')

        Y = torch.empty((N, L), dtype=torch.float32, device=dev)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
            Xc = Xb - mean32
            Y[start:end] = Xc @ Z_range

        Q, _ = torch.linalg.qr(Y, mode='reduced')

    # Compressed matrix B = Q.T @ Xc
    B = torch.zeros((L, M), dtype=torch.float32, device=dev)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
        Xc = Xb - mean32
        B += Q[start:end].T @ Xc

    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    S = S[:K].to(torch.float64)
    E = Vh[:K].T.contiguous().to(torch.float64)

    eigvals = (S.square() / N).clamp_min(1e-12)

    whitening = E / torch.sqrt(eigvals)[None, :]
    dewhitening = E * torch.sqrt(eigvals)[None, :]

    # Whitened data: Z = U * sqrt(N).
    # This avoids an additional pass computing Xc @ whitening.
    Z = (Q @ Ub[:, :K]) * np.sqrt(N)
    Z = Z.to(torch.float64)

    explained_ratio = S.square().sum().item() / max(norm_X_sq, 1e-12)

    print_and_log(f'PCA Explained Variance Ratio: {explained_ratio:.6f}')
    print_and_log(f'PCA Whitening Finished in {time.perf_counter() - start_time:.2f} s')
    print_and_log('')

    # --------------------------------------------------
    # 3. Symmetric FastICA fixed-point iteration
    # --------------------------------------------------
    Q_ica, _ = torch.linalg.qr(torch.randn(K, K, dtype=torch.float64, device=dev), mode='reduced')

    converged = False
    iteration_count = 0

    for iteration in range(iter_max):
        Y_ica = Z @ Q_ica.T

        # g(y) = tanh(y), g'(y) = 1 - tanh(y)^2
        G = torch.tanh(Y_ica)
        Gp = 1.0 - G.square()

        Q_new = (G.T @ Z) / N - Gp.mean(dim=0)[:, None] * Q_ica

        # Symmetric decorrelation: Q_new <- (Q_new Q_new.T)^(-1/2) Q_new
        U_ica, _, Vh_ica = torch.linalg.svd(Q_new, full_matrices=False)
        Q_new = U_ica @ Vh_ica

        lim = torch.max(torch.abs(torch.abs(torch.diagonal(Q_new @ Q_ica.T)) - 1)).item()

        Q_ica = Q_new
        iteration_count = iteration + 1

        if lim < tol:
            converged = True
            break

    if converged:
        print_and_log(f'FastICA converged at iteration {iteration_count} / {iter_max}.')
        print_and_log(f'lim = {lim:.6e} < tol = {tol}.')
    else:
        print_and_log(f'Warning: FastICA did not converge within {iter_max} iterations.')
        print_and_log(f'lim = {lim:.6e} > tol = {tol}.')

    # --------------------------------------------------
    # 4. Spatial sources and detector-space patterns
    # --------------------------------------------------
    W = Z @ Q_ica.T
    H = Q_ica @ dewhitening.T

    W32 = W.to(torch.float32)
    H32 = H.to(torch.float32)

    print_and_log('')
    print_and_log(f'ICA Total Process Finished in {time.perf_counter() - start_time:.2f} s')

    W = W32.T.reshape(K, sy, sx).cpu().numpy()
    H = H32.reshape(K, dy, dx).cpu().numpy()

    return W, H





# Perform PCA using covariance eigendecomposition.
# def PCA(data_4D, n_components, n_block=1, device='None'):

#     if device is None or device == 'None':
#         device = 'cpu'

#     print_and_log('')
#     print_and_log('##################### PCA Parameters ######################')
#     print_and_log(f'Device Using: {device}')
#     print_and_log(f'PCA Components Number: {n_components}')
#     print_and_log(f'PCA Block Number: {n_block}')

#     sy, sx, dy, dx = data_4D.shape

#     X_np = np.ascontiguousarray(data_4D.reshape(-1, dy * dx), dtype=np.float32)

#     N, M = X_np.shape
#     K = n_components

#     if not isinstance(n_block, int) or n_block < 1:
#         raise ValueError('n_block must be a positive integer.')

#     dev = torch.device(device)
#     block_size = (N + n_block - 1) // n_block

#     start_time = time.perf_counter()

#     # --------------------------------------------------
#     # 1. Compute mean in blocks
#     # --------------------------------------------------
#     mean = torch.zeros(M, dtype=torch.float64, device=dev)

#     for start in range(0, N, block_size):
#         Xb = torch.as_tensor(X_np[start:start + block_size], dtype=torch.float64, device=dev)
#         mean += Xb.sum(dim=0)
#     mean /= N

#     # --------------------------------------------------
#     # 2. Compute covariance matrix
#     # C = Xc.T @ Xc / N
#     # --------------------------------------------------
#     C = torch.zeros((M, M), dtype=torch.float64, device=dev)

#     for start in range(0, N, block_size):
#         Xb = torch.as_tensor(X_np[start:start + block_size], dtype=torch.float64, device=dev)
#         Xb = Xb - mean
#         C += Xb.T @ Xb

#     C /= N

#     # --------------------------------------------------
#     # 3. Eigenvalue decomposition
#     # --------------------------------------------------
#     eigvals, eigvecs = torch.linalg.eigh(C)

#     # Sort eigenvalues in descending order
#     idx = torch.argsort(eigvals, descending=True)[:K]

#     eigvals = eigvals[idx].clamp_min(0)
#     H = eigvecs[:, idx].T.contiguous()  # (K, M)

#     # Diagonal matrix of selected eigenvalues
#     Lambda = torch.diag(eigvals)  # (K, K)

#     # Explained variance ratio
#     explained_ratio = (eigvals.sum() / torch.trace(C).clamp_min(1e-12)).item()

#     # --------------------------------------------------
#     # 4. Compute spatial scores in blocks
#     # W = Xc @ H.T
#     # --------------------------------------------------
#     H32 = H.to(torch.float32)
#     mean32 = mean.to(torch.float32)

#     W = torch.empty((N, K), dtype=torch.float32, device=dev)

#     for start in range(0, N, block_size):
#         end = min(start + block_size, N)
#         Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
#         Xc = Xb - mean32
#         W[start:end] = Xc @ H32.T

#     # --------------------------------------------------
#     # 5. Reshape and return
#     # --------------------------------------------------
#     W = W.T.reshape(K, sy, sx).cpu().numpy()
#     H = H32.reshape(K, dy, dx).cpu().numpy()
#     Lambda = Lambda.cpu().numpy()

#     print_and_log(f'Explained Variance Ratio: {explained_ratio:.6f}')
#     print_and_log(f'PCA process finished in {time.perf_counter() - start_time:.2f} s')

#     return W, H, Lambda


# Perform FastICA with covariance-based PCA whitening.
# def fast_ICA(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, device='None'):

#     if device is None or device == 'None':
#         device = 'cpu'

#     print_and_log('')
#     print_and_log('##################### ICA Parameters ######################')
#     print_and_log(f'Device Using: {device}')
#     print_and_log(f'ICA Components Number: {n_components}')
#     print_and_log(f'Number of Iterations: {iter_max}')
#     print_and_log(f'ICA Convergence Tolerance: {tol}')
#     print_and_log(f'ICA Block Number: {n_block}')

#     sy, sx, dy, dx = data_4D.shape
#     X_np = np.ascontiguousarray(data_4D.reshape(-1, dy * dx), dtype=np.float32)

#     N, M = X_np.shape
#     K = n_components

#     dev = torch.device(device)
#     block_size = (N + n_block - 1) // n_block

#     # --------------------------------------------------
#     # 1. Mean and covariance
#     # --------------------------------------------------
#     mean = torch.zeros(M, dtype=torch.float64, device=dev)

#     for start in range(0, N, block_size):
#         Xb = torch.as_tensor(X_np[start:start + block_size], dtype=torch.float64, device=dev)
#         mean += Xb.sum(dim=0)

#     mean /= N

#     C = torch.zeros((M, M), dtype=torch.float64, device=dev)

#     for start in range(0, N, block_size):
#         Xb = torch.as_tensor(X_np[start:start + block_size], dtype=torch.float64, device=dev)
#         Xb -= mean
#         C += Xb.T @ Xb

#     C /= N
    
#     start_time = time.perf_counter()

#     # --------------------------------------------------
#     # 2. PCA whitening
#     # --------------------------------------------------
#     eigvals, eigvecs = torch.linalg.eigh(C)
#     idx = torch.argsort(eigvals, descending=True)[:K]

#     eigvals = eigvals[idx].clamp_min(1e-12)
#     E = eigvecs[:, idx]

#     # Z has unit covariance in retained PCA subspace
#     whitening = E / torch.sqrt(eigvals)[None, :]
#     dewhitening = E * torch.sqrt(eigvals)[None, :]
    
#     print_and_log(f'ICA process finished in {time.perf_counter() - start_time:.2f} s')

#     # --------------------------------------------------
#     # 3. Whitened data, stored in sample space
#     # --------------------------------------------------
#     Z = torch.empty((N, K), dtype=torch.float64, device=dev)

#     for start in range(0, N, block_size):
#         end = min(start + block_size, N)
#         Xb = torch.as_tensor(X_np[start:end], dtype=torch.float64, device=dev)
#         Z[start:end] = (Xb - mean) @ whitening

#     # --------------------------------------------------
#     # 4. Symmetric FastICA fixed-point iteration
#     # --------------------------------------------------
#     torch.manual_seed(0)
#     Q, _ = torch.linalg.qr(torch.randn(K, K, dtype=torch.float64, device=dev))

#     for iteration in range(iter_max):
#         Y = Z @ Q.T

#         # g(y) = tanh(y), g'(y) = 1 - tanh(y)^2
#         G = torch.tanh(Y)
#         Gp = 1.0 - G.square()

#         Q_new = (G.T @ Z) / N - (
#             Gp.mean(dim=0)[:, None] * Q
#         )

#         # Symmetric decorrelation:
#         # Q_new <- (Q_new Q_new.T)^(-1/2) Q_new
#         U, S, Vh = torch.linalg.svd(Q_new, full_matrices=False)
#         Q_new = U @ Vh

#         lim = torch.max(torch.abs(torch.abs(torch.diag(Q_new @ Q.T)) - 1)).item()

#         Q = Q_new

#         if lim < tol:
#             break

#     # if not converged:
#     #     print_and_log('Warning: FastICA did not converge.')

#     # --------------------------------------------------
#     # 5. Spatial sources and detector-space patterns
#     # --------------------------------------------------
#     W = Z @ Q.T
#     H = Q @ dewhitening.T

#     # Centered reconstruction: X - mean ≈ W @ H
#     norm_X_sq = 0.0
#     residual_sq = 0.0

#     mean32 = mean.to(torch.float32)
#     H32 = H.to(torch.float32)
#     W32 = W.to(torch.float32)

#     for start in range(0, N, block_size):
#         end = min(start + block_size, N)
#         Xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=dev)
#         Xc = Xb - mean32
#         residual = Xc - W32[start:end] @ H32

#         norm_X_sq += Xc.double().square().sum().item()
#         residual_sq += residual.double().square().sum().item()

#     err = np.sqrt(residual_sq / max(norm_X_sq, 1e-12))

#     print_and_log('')
#     print_and_log(f'FastICA Stoped at {iteration + 1}th Iteration')
#     print_and_log(f'ICA Relative Reconstruction Error: {err:.8e}')
#     print_and_log(f'ICA process finished in {time.perf_counter() - start_time:.2f} s')

#     W = W32.T.reshape(K, sy, sx).cpu().numpy()
#     H = H32.reshape(K, dy, dx).cpu().numpy()

#     return W, H




def nonnegative_independent_factorization(V, n_components=None, iter_max=1000, lr=1e-2, lambda_ind=0.1, tol=1e-7, device='cpu', random_state=0):
    import numpy as np
    import torch

    if device is None or device == 'None':
        device = 'cpu'

    V = torch.as_tensor(V, dtype=torch.float32, device=device)
    V = torch.clamp(V, min=0)

    N, M = V.shape
    R = min(n_components or M, N, M)

    if R < 1:
        raise ValueError('n_components must be >= 1.')

    eps = 1e-12
    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)

    # Positive initialization.
    S = torch.rand((N, R), generator=generator, device=device) + 0.1
    A = torch.rand((M, R), generator=generator, device=device) + 0.1
    S = S.requires_grad_()
    A = A.requires_grad_()

    optimizer = torch.optim.Adam([S, A], lr=lr)
    norm_V_sq = V.square().sum().clamp_min(eps)
    err_list = []
    previous_loss = None

    for iteration in range(iter_max):
        optimizer.zero_grad()

        V_hat = S @ A.T
        loss_recon = (V - V_hat).square().sum() / norm_V_sq

        # Correlation matrix of the nonnegative source components.
        S_centered = S - S.mean(dim=0, keepdim=True)
        S_std = S_centered.square().mean(dim=0).sqrt().clamp_min(1e-6)
        S_norm = S_centered / S_std
        corr = S_norm.T @ S_norm / N

        off_diag = corr - torch.diag_embed(torch.diagonal(corr))
        loss_ind = off_diag.square().sum() / max(R * (R - 1), 1)

        loss = loss_recon + lambda_ind * loss_ind
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # S.clamp_(min=eps)
            # A.clamp_(min=eps)

            # Fix the scale ambiguity without changing S @ A.T.
            scale = A.norm(dim=0).clamp_min(eps)
            A.div_(scale.unsqueeze(0))
            S.mul_(scale.unsqueeze(0))

            current_recon = (V - S @ A.T).square().sum() / norm_V_sq
            err = current_recon.sqrt().item()
            err_list.append(err)

            current_loss = loss.item()
            if previous_loss is not None:
                relative_change = abs(previous_loss - current_loss) / max(abs(previous_loss), eps)
                if relative_change < tol:
                    break
            previous_loss = current_loss

    with torch.no_grad():
        S_out = S.detach().cpu().numpy()
        A_out = A.detach().cpu().numpy()
        V_hat = S @ A.T
        final_err = ((V - V_hat).square().sum() / norm_V_sq).sqrt().item()
        corr_final = S_norm.T @ S_norm / N
        if R > 1:
            mask = ~torch.eye(R, dtype=torch.bool, device=device)
            mean_abs_corr = corr_final[mask].abs().mean().item()
        else:
            mean_abs_corr = 0.0

    print(f'Nonnegative independent factorization: iterations = {len(err_list)}, relative reconstruction error = {final_err:.6e}, mean |correlation| = {mean_abs_corr:.6e}')

    return S_out, A_out, err_list




def fast_hals_NMF_independent(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, device='None', independent_target='none', independent_components=None, independent_iter_max=1000, independent_lr=1e-2, lambda_ind=0.1, independent_tol=1e-7, random_state=0):

    if independent_target not in ('none', 'W', 'H', 'both'):
        raise ValueError("independent_target must be 'none', 'W', 'H', or 'both'.")

    # ----------------------------------------------------
    # Stage 1: NMF
    # ----------------------------------------------------
    W, H, err_list = fast_hals_NMF(data_4D, n_components, iter_max=iter_max, tol=tol, n_block=n_block, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, device=device)

    sy, sx, dy, dx = data_4D.shape
    K = W.shape[0]
    R = K if independent_components is None else independent_components

    if independent_target == 'none':
        return W, H, err_list

    if R < 1:
        raise ValueError('independent_components must be >= 1.')

    dev = 'cpu' if device is None or device == 'None' else device

    # W: (K, sy, sx) -> (sy*sx, K)
    W_mat = np.ascontiguousarray(W.reshape(K, -1).T, dtype=np.float32)

    # H: (K, dy, dx) -> (dy*dx, K)
    H_mat = np.ascontiguousarray(H.reshape(K, -1).T, dtype=np.float32)

    # ----------------------------------------------------
    # Stage 2A: Nonnegative independent spatial components
    # W_mat ~= S @ A.T
    # ----------------------------------------------------
    if independent_target in ('W', 'both'):

        S, A, _ = nonnegative_independent_factorization(W_mat, n_components=R, iter_max=independent_iter_max, lr=independent_lr, lambda_ind=lambda_ind, tol=independent_tol, device=dev, random_state=random_state)

        # W_mat ~= S @ A.T
        # X ~= S @ A.T @ H_mat.T
        # Absorb A into the diffraction factors.
        H_mat = H_mat @ A
        W_mat = S

        K = R

    # ----------------------------------------------------
    # Stage 2B: Nonnegative independent diffraction components
    # H_mat ~= S @ A.T
    # ----------------------------------------------------
    if independent_target in ('H', 'both'):

        S, A, _ = nonnegative_independent_factorization(H_mat, n_components=R, iter_max=independent_iter_max, lr=independent_lr, lambda_ind=lambda_ind, tol=independent_tol, device=dev, random_state=random_state + 1)

        # H_mat ~= S @ A.T
        # X ~= W_mat @ A @ S.T
        W_mat = W_mat @ A
        H_mat = S

        K = R

    # ----------------------------------------------------
    # Stage 3: Restore original output convention
    # ----------------------------------------------------
    W_out = np.ascontiguousarray(W_mat.T.reshape(W_mat.shape[1], sy, sx), dtype=np.float32)
    H_out = np.ascontiguousarray(H_mat.T.reshape(H_mat.shape[1], dy, dx), dtype=np.float32)

    print_and_log('')
    print_and_log('################ Nonnegative Independent Factorization ################')
    print_and_log(f'Independent Target: {independent_target}')
    print_and_log(f'Output Components: {W_out.shape[0]}')
    print_and_log(f'W Output Shape: {W_out.shape}')
    print_and_log(f'H Output Shape: {H_out.shape}')

    return W_out, H_out, err_list
