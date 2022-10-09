# -*- coding: utf-8 -*-
""" 
Provides functions to estimate the parameters of a multi singleparameter
beta distribution mixture model (MBM) with the spectral method of moments.

@author: Martin Gölz
"""

import numpy as np
from scipy import stats
from scipy import linalg
import warnings

def learnGMM(x, N, d, k, reps_eta, gaussian_eta, var=True):
    """Learn the parameters of a multivariate Gaussian mixture by the spectral
    method of moments.
    """
    # % Define locally needed functions # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def third_moment_as_function(T, eta, d):
        # T: a third-order tensor of dim d x d x d
        # eta: a d x 1 vector
        # T(eta): The result is a d x d vector
        res = np.zeros((d, d))
        for i_1 in np.arange(0, d, 1):
            e_i_1 = np.zeros(d)
            e_i_1[i_1] = 1
            for i_2 in np.arange(0, d, 1):
                e_i_2 = np.zeros(d)
                e_i_2[i_2] = 1
                for i_3 in np.arange(0, d, 1):
                    res = res + T[i_1, i_2, i_3]*eta[i_3]*np.outer(e_i_1, e_i_2)
        return res

    def third_order_tensor(Y, U, V, W):
        # Y is of dimension m x m x m
        # U, V and W are of dimension m x n
        # The output is of dimension n x n x n
        m = Y.shape[0]
        n = U.shape[1]
        res = np.zeros((n, n, n))
        for j_1 in np.arange(0, n, 1):
            for j_2 in np.arange(0, n, 1):
                for j_3 in np.arange(0, n, 1):              
                    for i_1 in np.arange(0, m, 1):
                        for i_2 in np.arange(0, m, 1):
                            for i_3 in np.arange(0, m, 1):
                                res[j_1, j_2, j_3] = (
                                    res[j_1, j_2, j_3] +
                                    U[i_1, j_1]*V[i_2, j_2]*W[i_3, j_3]
                                    * Y[i_1, i_2, i_3])
        del j_1, j_2, j_3, i_1, i_2, i_3, m,n
        return res
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # If var is true, we estimate the variance also assuming different
    # variances for each component. If var is false, the variance is estimated
    # assuming similar variances for each components.

    # Splitting the input data into two sets. The data is not permuted here!
    # Must be permuted upfront, if desired.
    S_idx = np.arange(0, int(N/2))
    S_usc_idx = np.arange(int(N/2), N)

    # Empirical mean value
    mu_hat = 1/np.size(S_idx)*np.sum(x[S_idx, :], axis=0)
    mu_usc_hat = 1/np.size(S_usc_idx)*np.sum(x[S_usc_idx, :], axis=0)

    # # Empirical covariance matrix of the first set
    # cov_mat = np.zeros((d, d))
    # for N_idx in S_idx:
    #     cov_mat = (cov_mat
    #                + np.outer(x[N_idx, :] - cov_mat,
    #                           x[N_idx, :] - cov_mat)
    #                )
    # del N_idx
    # cov_mat = 1/np.size(S_idx) * cov_mat
    
    # # The average variance, the smallest eigenvalue of the covariance matrix
    # cov_mat_eig_val, cov_mat_eig_vec = np.linalg.eig(cov_mat)
    # idx_sig_sq_ol_var = np.argmin(emp_cor_eig_val_var)
    # sig_sq_ol_var_hat = emp_cor_eig_val_var[idx_sig_sq_ol_var]
    # v_var = emp_cor_eig_vec[:, idx_sig_sq_ol]

    # Empirical second moments
    M_2_hat_it = np.zeros((np.size(S_idx), d, d))
    for (cnt_idx, s_idx) in enumerate(S_idx):
        M_2_hat_it[cnt_idx, :, :] = np.outer(x[s_idx, :], x[s_idx, :])
    M_2_hat = 1/np.size(S_idx)*np.sum(M_2_hat_it, axis=0)
    del M_2_hat_it, cnt_idx, s_idx

    M_2_hat_usc_it = np.zeros((np.size(S_usc_idx), d, d))
    for (cnt_idx, s_idx) in enumerate(S_usc_idx):
        M_2_hat_usc_it[cnt_idx, :, :] = np.outer(x[s_idx, :], x[s_idx, :])
    M_2_usc_hat = 1/np.size(S_usc_idx)*np.sum(M_2_hat_usc_it, axis=0)
    del M_2_hat_usc_it, cnt_idx, s_idx

    # Empirical third moments
    M_3_hat_it = np.zeros((np.size(S_usc_idx), d, d, d))
    for (cnt_idx, s_idx) in enumerate(S_usc_idx):
        M_3_hat_it[cnt_idx, :, :, :] = np.tensordot(
            np.outer(x[s_idx, :], x[s_idx, :]), x[s_idx, :], axes=0)
    M_3_hat = 1/np.size(S_usc_idx)*np.sum(M_3_hat_it, axis=0)
    del M_3_hat_it, cnt_idx, s_idx

    # LearnGMM - Step 2
    cov_mat = M_2_hat - np.outer(mu_hat, mu_hat)
    (cov_mat_eig_val, cov_mat_eig_vec) = np.linalg.eig(cov_mat)
    # sig_sq_ol_hat_idx = np.argsort(-cov_mat_eig_val)[k-1]
    sig_sq_ol_hat_idx = np.argsort(cov_mat_eig_val)[0]

    # # #
    # Estimation of the number of correlated componets by MDL
    # Does not work because the largest EV is very dominant.
    #
    # First code snipped by Visa -> Preferred over the second snippet
    # (k, sig_sq_ol_hat) = learnNumMixCmpAndNseVarByMDLAIC(
    #    cov_mat_eig_val, np.size(S_idx))
    # Second code snippet by Visa
    # (k, sig_sq_ol_hat) = learnNumMixCmpAndNseVarByMDL(
    #     cov_mat_eig_val, np.size(S_idx))
    # # #
    sig_sq_ol_hat = cov_mat_eig_val[sig_sq_ol_hat_idx]
    v = cov_mat_eig_vec[:, sig_sq_ol_hat_idx]

    # LearnGMM - Repeat Step 2 for USC
    cov_mat_usc = M_2_usc_hat - np.outer(mu_usc_hat, mu_usc_hat)
    (cov_mat_usc_eig_val, cov_mat_usc_eig_vec) = np.linalg.eig(cov_mat_usc)
    # sig_sq_ol_hat_usc_idx = np.argsort(-cov_mat_usc_eig_val)[k]
    sig_sq_ol_hat_usc_idx = np.argsort(cov_mat_usc_eig_val)[0]
    sig_sq_ol_hat_usc = cov_mat_usc_eig_val[sig_sq_ol_hat_usc_idx]
    v_usc = cov_mat_usc_eig_vec[:, sig_sq_ol_hat_usc_idx]

    # Empirical first moment
    M_1_hat_it = np.zeros((np.size(S_idx), d))
    for (cnt_idx, s_idx) in enumerate(S_idx):
        M_1_hat_it[cnt_idx, :] = (
            x[s_idx, :]*(np.dot(v, (x[s_idx, :] - mu_hat))**2)
            )
    M_1_hat = 1/np.size(S_idx)*np.sum(M_1_hat_it, axis=0)
    del M_1_hat_it, cnt_idx, s_idx

    M_1_hat_usc_it = np.zeros((np.size(S_usc_idx), d))
    for (cnt_idx, s_idx) in enumerate(S_usc_idx):
        M_1_hat_usc_it[cnt_idx, :] = (
            x[s_idx, :]*(np.dot(v_usc, (x[s_idx, :] - mu_usc_hat))**2)
            )
    M_1_hat_usc = 1/np.size(S_usc_idx)*np.sum(M_1_hat_usc_it, axis=0)
    del M_1_hat_usc_it, cnt_idx, s_idx

    # LearnGMM - Step 3
    (U_M_2, s_M_2, V_T_M_2) = np.linalg.svd(
        M_2_hat - sig_sq_ol_hat*np.eye(d, d), full_matrices=False)
    sel_idx = np.arange(k)
    S_M_2 = np.real(np.diag(s_M_2[sel_idx]))
    U_M_2 = np.real(U_M_2[:, sel_idx])
    V_M_2 = np.real(np.transpose(V_T_M_2[sel_idx, :]))
    del s_M_2, V_T_M_2, sel_idx

    # Low rank approximation
    M_2_hat_lr = np.dot(U_M_2, np.dot(S_M_2, np.transpose(V_M_2)))

    # LearnGMM - Step 4
    U_hat, sgl_val, _ = np.linalg.svd(M_2_hat_lr, full_matrices=False)
    sel_idx = np.argsort(-np.real(sgl_val))[0:k]
    U_hat = np.real(U_hat[:, sel_idx])
    del sgl_val, sel_idx

    # LearnGMM - Step 5
    prd_fac = np.dot(np.transpose(U_hat), np.dot(M_2_hat_lr, U_hat))
    W_hat = np.dot(U_hat, np.real(linalg.sqrtm(np.linalg.pinv(prd_fac))))
    B_hat = np.dot(U_hat, np.real(linalg.sqrtm(prd_fac)))
    del prd_fac


    # LearnGMM - Step 6
    if gaussian_eta:
        # LearnGMM - Step 7
        sub_term = np.zeros((d, d, d, d))
        for i in np.arange(0, d, 1):
            e_i = np.zeros(d)
            e_i[i] = 1
            sub_term[i, :, :, :] = (
                np.tensordot(
                    np.tensordot(M_1_hat_usc, e_i, axes=0), e_i, axes=0)
                + np.tensordot(
                    np.tensordot(e_i, M_1_hat_usc, axes=0), e_i, axes=0)
                + np.tensordot(
                    np.tensordot(e_i, e_i, axes=0), M_1_hat_usc, axes=0)
                )
        M_3_tensor_hat = M_3_hat - np.sum(sub_term, axis=0)
    else:
        wht_M_1 = np.dot(np.transpose(W_hat), M_1_hat_usc)
        if var == False:
            wht_M_1 = np.dot(np.transpose(W_hat), sig_sq_ol_hat*mu_hat)
        wht_M_3 = third_order_tensor(M_3_hat, W_hat, W_hat, W_hat)
    
        # LearnGMM - Step 7
        sub_term = np.zeros((d, k, k, k))
        for i in np.arange(0, d, 1):
            e_i = np.zeros(d)
            e_i[i] = 1
            wht_e_i = np.dot(np.transpose(W_hat), e_i)
            sub_term[i, :, :, :] = (
                np.tensordot(
                    np.tensordot(wht_M_1, wht_e_i, axes=0), wht_e_i, axes=0)
                + np.tensordot(
                    np.tensordot(wht_e_i, wht_M_1, axes=0), wht_e_i, axes=0)
                + np.tensordot(
                    np.tensordot(wht_e_i, wht_e_i, axes=0), wht_M_1, axes=0)
                )
        M_3_gmm_hat = wht_M_3 - np.sum(sub_term, axis=0)

    # LearnGMM - Step 8
    delta = 1e-6
    t = np.ceil(np.log2(1/delta)).astype(int)
    if not gaussian_eta:
        theta = np.zeros((t, k))
        M_3_gmm_hat_theta = np.zeros((t, k, k))
        lam_hat = np.zeros((t, k))
        v_hat = np.zeros((t, k, k))
        test_stat = np.zeros(t)
        for t_idx in np.arange(0, t, 1):
            # Generating the random thehta on the unit surface
            theta[t_idx, :] = stats.norm.rvs(size=k)
            theta[t_idx, :] = (theta[t_idx, :]/np.sqrt(np.sum(theta[t_idx, :]**2)))
            # Computation of the matrix we get eigenvalues and eigenvectors from
            M_3_gmm_hat_theta[t_idx, :, :] = third_moment_as_function(
                M_3_gmm_hat, theta[t_idx, :], k)
            # LearnGMM - Step 8.b
            (val, vec) = np.linalg.eig(M_3_gmm_hat_theta[t_idx, :, :])
            lam_hat[t_idx, :] = np.real(val)
            v_hat[t_idx, :, :] = np.real(vec)
            del val, vec
            aux = np.zeros((k, k))
            for row_idx in np.arange(0, k, 1):
                for col_idx in np.arange(0, k, 1):
                    aux[row_idx, col_idx] = np.abs(
                            lam_hat[t_idx, row_idx] - lam_hat[t_idx, col_idx])
            if k == 1:
                test_stat[t_idx] = np.abs(lam_hat[t_idx])
            else:
                test_stat[t_idx] = (np.min([np.min(aux[~np.eye(k, dtype=bool)]),
                                            np.min(np.abs(lam_hat[t_idx, :]))]))
            del aux

            sel_idx = np.argmax(test_stat)

            # LearnGMM - Step 9
            mu_i_hat = np.zeros((k, d))
            for k_idx in np.arange(0, k, 1):
                mu_i_hat[k_idx, :] = np.dot(
                    lam_hat[sel_idx, k_idx]/np.dot(
                        theta[sel_idx, :], v_hat[sel_idx, :, k_idx]),
                    np.dot(B_hat, v_hat[sel_idx, :, k_idx])
                    )        
            w_hat = np.dot(np.transpose(np.linalg.pinv(mu_i_hat)), mu_hat)
            sig_sq_hat = np.zeros(k)
            for k_idx in np.arange(0, k, 1):
                sig_sq_hat[k_idx] = 1/w_hat[k_idx]*np.dot(
                    np.transpose(np.linalg.pinv(mu_i_hat)), M_1_hat)[k_idx]
            sig_sq_hat_usc = np.zeros(k)
            for k_idx in np.arange(0, k, 1):
                sig_sq_hat_usc[k_idx] = 1/w_hat[k_idx]*np.dot(
                    np.transpose(np.linalg.pinv(mu_i_hat)), M_1_hat_usc)[k_idx]
            if var is False:
                sig_sq_hat = sig_sq_ol_hat * np.ones(k)

            reps_eta = 1
    else:
        t = reps_eta#0
        eta = np.zeros((t, d))
        M_3_gmm_hat_eta = np.zeros((t, k, k))
        lam_hat = np.zeros((t, k))
        v_hat = np.zeros((t, k, k))
        test_stat = np.zeros(t)
        mu_i_hat = np.zeros((t, k, d))
        w_hat = np.zeros((t, k))
        sig_sq_hat = np.zeros((t, k))
        for t_idx in np.arange(t):
            eta[t_idx, :] = stats.norm.rvs(size=d)
            eta[t_idx, :] = eta[t_idx, :]/np.sqrt(eta[t_idx, :]**2)
            # for j1 in np.arange(k):
            #     for j2 in np.arange(k):
            #         for i1 in np.arange(d):
            #             for i2 in np.arange(d):
            #                 M_3_gmm_hat_eta[t_idx, j1, j2] += W_hat[i1, j1]*W_hat[i2, j2]*np.sum(M_3_tensor_hat[i1, i2, :]*eta[t_idx, :])
            M_3_gmm_hat_eta[t_idx, :, :] = np.dot(np.transpose(W_hat), np.dot(np.sum(M_3_tensor_hat*eta[t_idx, :], axis=2), W_hat))
            (val, vec) = np.linalg.eig(M_3_gmm_hat_eta[t_idx, :, :])
            lam_hat[t_idx, :] = np.real(val)
            v_hat[t_idx, :, :] = np.real(vec)
            # del val, vec
            # aux = np.zeros((k, k))
            # for row_idx in np.arange(0, k, 1):
            #     for col_idx in np.arange(0, k, 1):
            #         aux[row_idx, col_idx] = np.abs(
            #                 lam_hat[t_idx, row_idx] - lam_hat[t_idx, col_idx])
            # if k == 1:
            #     test_stat[t_idx] = np.abs(lam_hat[t_idx])
            # else:
            #     test_stat[t_idx] = (np.min([np.min(aux[~np.eye(k, dtype=bool)]),
            #                                 np.min(np.abs(lam_hat[t_idx, :]))]))
            # del aux
            for k_idx in np.arange(0, k, 1):
                mu_i_hat[t_idx, k_idx, :] = np.dot(
                    lam_hat[t_idx, k_idx]/np.dot(
                        np.dot(np.transpose(eta[t_idx, :]), B_hat), v_hat[t_idx, :, k_idx]),
                    np.dot(B_hat, v_hat[t_idx, :, k_idx])
                    )
            w_hat[t_idx, :] = np.dot(np.transpose(np.linalg.pinv(mu_i_hat[t_idx, :, :])), mu_hat)
            for k_idx in np.arange(0, k, 1):
                sig_sq_hat[t_idx, k_idx] = 1/w_hat[t_idx, k_idx]*np.dot(
                    np.transpose(np.linalg.pinv(mu_i_hat[t_idx, :, :])), M_1_hat)[k_idx]
    # del sub_term, i, wht_M_1, wht_M_3
            reps_eta = t

    return w_hat, mu_i_hat, sig_sq_hat#, sig_sq_hat_usc


def learnMBM(x, N, d, k, rge, reps_eta, gaussian_eta=True):
    """Learn the parameters of a multivariate multi-single-parameter beta
    mixture by the spectral method of moments.
    """
    # % Define locally needed functions # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def is_d_ok(a_run, b_run):
        a_int_max = 10
        a_int_min = 0
        b_int_max = 10
        b_int_min = 0
        return (a_int_max >= a_run and
                a_int_min <= a_run and
                b_int_max >= b_run and
                b_int_min <= b_run)


    def is_k_ok(w_hat):
        use_k_idc = np.all([w_hat >= 0], axis=0)
        return use_k_idc


    def get_pdf_vls(dat, a, b, w, d, k):
        pdf = np.zeros((k, d, dat.size))
        use_k_idc = is_k_ok(w)
        use_d_idc = np.ones(a.shape)
        use_d_idc[~use_k_idc] = 0
        for k_idx in np.arange(0, k, 1):
            if use_k_idc[k_idx]:
                for d_idx in np.arange(0, d, 1):
                    use_d_idc[k_idx, d_idx] = is_d_ok(
                        a[k_idx, d_idx], b[k_idx, d_idx])
                    if use_d_idc[k_idx, d_idx]:
                        pdf[k_idx, d_idx, :] = (
                            stats.beta.pdf(
                                dat, a[k_idx, d_idx], b[k_idx, d_idx]))
                    else:
                        pdf[k_idx, d_idx, :] = np.nan
            else:
                pdf[k_idx, :, :] = np.nan
                use_d_idc[k_idx, :] = 0
        use_k_idc[np.where(use_k_idc)[0]] = np.sum(
            use_d_idc[np.where(use_k_idc)[0], :], 1) > 0
        pdf_av = np.zeros((d, dat.size))
        w_tmp = np.copy(w)
        w_tmp[~use_k_idc] = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_tmp = w_tmp/np.sum(w_tmp)
        for k_idx in np.where(use_k_idc)[0]:
            pdf_av[np.where(
                use_d_idc[k_idx, :])[0], :] = (pdf_av[np.where(
                    use_d_idc[k_idx, :])[0], :] + w_tmp[k_idx] *
                    1/np.sum(use_d_idc[k_idx, :]) * pdf[k_idx, np.where(
                        use_d_idc[k_idx, :])[0], :])
        # pdf_av = np.mean(pdf_av, axis=0)
        pdf_av = np.sum(pdf_av, axis=0)
        return pdf_av

    def get_cdf_vls(dat, a, b, w, d, k):
        cdf = np.zeros((k, d, dat.size))
        use_k_idc = is_k_ok(w)
        use_d_idc = np.ones(a.shape)
        use_d_idc[~use_k_idc] = 0
        for k_idx in np.arange(0, k, 1):
            if use_k_idc[k_idx]:
                for d_idx in np.arange(0, d, 1):
                    use_d_idc[k_idx, d_idx] = is_d_ok(
                        a[k_idx, d_idx], b[k_idx, d_idx])
                    if use_d_idc[k_idx, d_idx]:
                        cdf[k_idx, d_idx, :] = (stats.beta.cdf(
                                dat, a[k_idx, d_idx], b[k_idx, d_idx]))
                    else:
                        cdf[k_idx, d_idx, :] = np.nan
            else:
                cdf[k_idx, :, :] = np.nan
                use_d_idc[k_idx, :] = 0
        use_k_idc[np.where(use_k_idc)[0]] = np.sum(
            use_d_idc[np.where(use_k_idc)[0], :], 1)>0
        cdf_av = np.zeros((d, dat.size))
        w_tmp = np.copy(w)
        w_tmp[~use_k_idc] = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_tmp = w_tmp/np.sum(w_tmp)
        for k_idx in np.where(use_k_idc)[0]:
            cdf_av[np.where(
                use_d_idc[k_idx, :])[0], :] = (cdf_av[np.where(
                use_d_idc[k_idx, :])[0], :]  + w_tmp[k_idx] *
                    1/np.sum(use_d_idc[k_idx, :]) * cdf[k_idx, np.where(
                use_d_idc[k_idx, :])[0], :]) 
        # cdf_av = np.mean(cdf_av, axis=0)
        cdf_av = np.sum(cdf_av, axis=0)
        return cdf_av
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    w_hat, mu_hat, sig_sq_hat = learnGMM(x, N, d, k, reps_eta, gaussian_eta)
    sig_sq_hat = np.tile(sig_sq_hat, [k])
    a_hat = np.zeros((reps_eta, k, d))
    b_hat = np.ones((reps_eta, k, d))
    for t_idx in np.arange(reps_eta):
        for k_idx in np.arange(0, k, 1):
            a_hat[t_idx, k_idx, :] = mu_hat[t_idx, k_idx, :]/(
                1-mu_hat[t_idx, k_idx, :])

    beta_pdf_hat_av = np.zeros((reps_eta, rge.size))
    beta_cdf_hat_av = np.zeros((reps_eta, rge.size))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for t_idx in np.arange(reps_eta):
            beta_pdf_hat_av[t_idx, :] = get_pdf_vls(
                rge, a_hat[t_idx, :, :], b_hat[t_idx, :, :],
                w_hat[t_idx, :], d, k)
            beta_cdf_hat_av[t_idx, :] = get_cdf_vls(
                rge, a_hat[t_idx, :, :], b_hat[t_idx, :, :],
                w_hat[t_idx, :], d, k)

    return a_hat, b_hat, w_hat, beta_pdf_hat_av, beta_cdf_hat_av

