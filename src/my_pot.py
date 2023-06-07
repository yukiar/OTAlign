import torch


def entropic_partial_wasserstein(a, b, M, reg, m=None, numItermax=1000,
                                 stopThr=1e-100, verbose=False, log=False):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    M = M.to(torch.float64)

    dim_a, dim_b = M.shape
    dx = torch.ones(dim_a, dtype=torch.float64, device='cuda')
    dy = torch.ones(dim_b, dtype=torch.float64, device='cuda')

    if len(a) == 0:
        a = torch.ones(dim_a, dtype=torch.float64, device='cuda') / dim_a
    if len(b) == 0:
        b = torch.ones(dim_b, dtype=torch.float64, device='cuda') / dim_b

    if m is None:
        m = torch.minimum(torch.sum(a.detach()), torch.sum(b.detach())) * 1.0
    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > torch.minimum(torch.sum(a.detach()), torch.sum(b.detach())):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    K = torch.exp(-M / reg)
    K = K * m / torch.sum(K)

    err, cpt = 1, 0
    q1 = torch.ones(K.shape, device='cuda')
    q2 = torch.ones(K.shape, device='cuda')
    q3 = torch.ones(K.shape, device='cuda')

    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K = K * q1
        K1 = torch.matmul(torch.diag(torch.minimum(a / torch.sum(K, axis=1), dx)), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        K2 = torch.matmul(K1, torch.diag(torch.minimum(b / torch.sum(K1, axis=0), dy)))
        q2 = q2 * K1prev / K2
        K2prev = K2
        K2 = K2 * q3
        K = K2 * (m / torch.sum(K2))
        q3 = q3 * K2prev / K

        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
        if cpt % 10 == 0:
            err = torch.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    log_e['partial_w_dist'] = torch.sum(M * K)
    if log:
        return K, log_e
    else:
        return K


def sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m_1, reg_m_2, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, **kwargs):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    M = M.to(torch.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = torch.ones(dim_a, dtype=torch.float64, device='cuda') / dim_a
    if len(b) == 0:
        b = torch.ones(dim_b, dtype=torch.float64, device='cuda') / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = torch.ones((dim_a, 1), dtype=torch.float64, device='cuda') / dim_a
        v = torch.ones((dim_b, n_hists), dtype=torch.float64, device='cuda') / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = torch.ones(dim_a, dtype=torch.float64, device='cuda') / dim_a
        v = torch.ones(dim_b, dtype=torch.float64, device='cuda') / dim_b

    K = torch.exp(M / (-reg))

    fi_1 = reg_m_1 / (reg_m_1 + reg)
    fi_2 = reg_m_2 / (reg_m_2 + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = torch.mv(K, v)
        u = (a / Kv) ** fi_1
        Ktu = torch.mv(K.T, u)
        v = (b / Ktu) ** fi_2

        if (torch.any(Ktu == 0.)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = torch.max(torch.abs(u - uprev)) / max(
            torch.max(torch.abs(u)), torch.max(torch.abs(uprev)), 1.
        )
        err_v = torch.max(torch.abs(v - vprev)) / max(
            torch.max(torch.abs(v)), torch.max(torch.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = torch.log(u + 1e-300)
        log['logv'] = torch.log(v + 1e-300)

    if n_hists:  # return only loss
        res = torch.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        P = u[:, None] * K * v[None, :]
        P = P.to(torch.float32)
        if log:
            return P, log
        else:
            return P
