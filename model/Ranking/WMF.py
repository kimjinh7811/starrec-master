import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import sparse as sp

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from scipy.sparse.linalg import spsolve

class WMF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(WMF, self).__init__(dataset, model_conf)
        device = torch.device("cpu")
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.embedding_dim = model_conf['embedding_dim']
        self.alpha = model_conf['alpha']
        self.lambda_U = model_conf['lambda_u']
        # self.lambda_V = model_conf['lambda_v']
        self.lambda_V = self.lambda_U

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.eps = 1e-6
        self.device = device

        self.build_graph()

    def build_graph(self):
        train_matrix = self.dataset.train_matrix
        self.S = self.linear_surplus_confidence_matrix(train_matrix, self.alpha)

        self.U = None
        # if not fixed_item_embeddings and not V:
        self.V = np.random.randn(self.num_items, self.embedding_dim).astype('float32') * 0.01

        self.set_recompute_solve_functions()

    def forward(self, user_ids):
        X_pred = self.U[user_ids].dot(self.V.T)
        return X_pred

    def linear_surplus_confidence_matrix(self, B, alpha):
        # To construct the surplus confidence matrix, we need to operate only on
        # the nonzero elements.
        # This is not possible: S = alpha * B
        S = B.copy()
        S.data = alpha * S.data
        return S

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        """
        Train model following given config.

        """
        # exp conf.
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        
        # for each epoch
        start = time.time()
        users = torch.arange(self.num_users).long().to(self.device)
        items = torch.arange(self.num_items).long().to(self.device)

        if verbose:
            print("Precompute S^T (if necessary)")
            start_time = time.time()
        
        ST = self.S.T.tocsr()

        if verbose:
            print("  took %.3f seconds" % (time.time() - start_time))
            start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_train_start = time.time()
            
            # def recompute_factors(self, Y, S, lambda_reg, X, batch_size, n_jobs=1):            
            self.U = self.recompute_func(self.V, self.S, self.lambda_U, self.batch_size, self.solve_func)
            
            self.V = self.recompute_func(self.U, ST, self.lambda_V, self.batch_size, self.solve_func)

            epoch_train_time = time.time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch, 'train time=%.2f' % epoch_train_time]

            # evaluate on valid data
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time.time()
                
                valid_score = evaluator.evaluate(self, dataset)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p')) 

                epoch_eval_time = time.time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))
        
        total_train_time = time.time() - start

        return early_stop.best_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        eval_output = self.forward(user_ids)
        if eval_items is not None:
            eval_output[np.logical_not(eval_items)]=float('-inf')
        return eval_output

    def state_dict(self):
        return {'U': self.U, 'V': self.V}

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state = torch.load(f)
        self.U = state['U']
        self.V = state['V']

    def set_recompute_solve_functions(self):
        # recompute func
        self.recompute_func = self.recompute_factors_bias_batched_precompute

        # solve_func
        try:
            # Necessary packages for gpu solver
            import pycuda.gpuarray
            import pycuda.autoinit
            import scikits
            import scikits.cuda
            from scikits.cuda import linalg
            from scikits.cuda import cublas
            self.solve_func = self.solve_gpu
        except Exception as e:
            print(e)
            print('Pycuda and scikit-cuda cannot be loaded. Solve sequentially.')
            self.solve_func = self.solve_sequential

    ##################################################################
    ##   recompute_factors functions                                ##
    ##################################################################
    def recompute_factors_bias_batched_precompute(self, Y, S, lambda_reg, batch_size=1, solve=None):
        """
        Like recompute_factors_bias_batched, but doing a bunch of batch stuff outside the for loop.
        """
        m = S.shape[0] # m = number of users
        f = Y.shape[1] # f = number of factors
        # f = Y.shape[1] - 1 # f = number of factors
        
        # b_y = Y[:, f] # vector of biases
        Y_e = Y.copy()
        # Y_e[:, f] = 1 # factors with added column of ones
        YTY = np.dot(Y_e.T, Y_e) # precompute this

        # R = np.eye(f + 1) # regularization matrix
        R = np.eye(f) # regularization matrix
        # R[f, f] = 0 # don't regularize the biases!
        R *= lambda_reg

        YTYpR = YTY + R

        # byY = np.dot(b_y, Y_e) # precompute this as well

        X_new = np.zeros((m, f), dtype='float32')

        num_batches = int(np.ceil(m / float(batch_size)))

        rows_gen = iter_rows(S)

        for b in range(num_batches):
            lo = b * batch_size
            hi = min((b + 1) * batch_size, m)
            current_batch_size = hi - lo

            lo_batch = S.indptr[lo]
            hi_batch = S.indptr[hi] # hi - 1 + 1

            i_batch = S.indices[lo_batch:hi_batch]
            s_batch = S.data[lo_batch:hi_batch]
            Y_e_batch = Y_e[i_batch]
            # b_y_batch = b_y[i_batch]

            # precompute the left hand side of the dot product for computing A for the entire batch.
            a_lhs_batch = s_batch + 1
            # a_lhs_batch = (1 - b_y_batch) * s_batch + 1

            # also precompute the right hand side of the dot product for computing B for the entire batch.
            b_rhs_batch = Y_e_batch * s_batch[:, None]

            # A_stack = np.empty((current_batch_size, f + 1), dtype='float32')
            # B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype='float32')

            A_stack = np.empty((current_batch_size, f), dtype='float32')
            B_stack = np.empty((current_batch_size, f, f), dtype='float32')

            for k in range(lo, hi):
                ib = k - lo # index inside the batch

                lo_iter = S.indptr[k] - lo_batch
                hi_iter = S.indptr[k + 1] - lo_batch

                s_u = s_batch[lo_iter:hi_iter]
                Y_u = Y_e_batch[lo_iter:hi_iter]
                a_lhs_u = a_lhs_batch[lo_iter:hi_iter]
                b_rhs_u = b_rhs_batch[lo_iter:hi_iter]

                A_stack[ib] = np.dot(a_lhs_u, Y_u)
                B_stack[ib] = np.dot(Y_u.T, b_rhs_u)

            # A_stack -= byY[None, :]
            B_stack += YTYpR[None, :, :]

            # print("start batch solve %d" % b)
            X_stack = solve(A_stack, B_stack)
            # print("finished")
            X_new[lo:hi] = X_stack

        return X_new

    ##################################################################
    ##   solve_batch functions                                      ##
    ##################################################################
    # 1. BASE
    def solve_sequential(self, As, Bs):
        X_stack = np.empty_like(As, dtype=As.dtype)

        for k in range(As.shape[0]):
            X_stack[k] = np.linalg.solve(Bs[k], As[k])

        return X_stack

    # # 2. Multiprocessing
    # def process_func(self, tup):
    #     A, B = tup
    #     return np.linalg.solve(B.T, A.T).T

    # def solve_mp(self, As, Bs, NUM_PROCESSES=4):
    #     try:
    #         import multiprocessing as mp
    #     except:
    #         print()
    #         return self.solve_sequential(As, Bs)

    #     pool = mp.Pool(NUM_PROCESSES)
    #     X_list = pool.map(process_func, zip(As, Bs))
    #     pool.close() # IMPORTANT!
    #     Xs = np.array(X_list)
    #     return Xs
    
    def solve_gpu(self, As, Bs):
        # solver gpu
        try:
            import pycuda.gpuarray
            import pycuda.autoinit
            import scikits.cuda
            from scikits.cuda import linalg
            from scikits.cuda import cublas
        except:
            print('asdfasdf')
            return self.solve_sequential(As, Bs)

        linalg.init()
        def bptrs(a):
            """
            Pointer array when input represents a batch of matrices or vectors.
            taken from scikits.cuda tests/test_cublas.py
            """
            
            return pycuda.gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                        dtype=cublas.ctypes.c_void_p)


        allocated_shape = [None]
        allocations = [None]

        batch_size, num_factors = As.shape

        if allocated_shape[0] == As.shape: # reuse previous allocations
            As_gpu, Bs_gpu, P_gpu, info_gpu, Cs_gpu, Rs_gpu, A_arr, B_arr, C_arr, R_arr = allocations[0]
            As_gpu.set(As)
            Bs_gpu.set(Bs)
        else: # allocate
            # transfer As and Bs to GPU
            As_gpu = pycuda.gpuarray.to_gpu(As.astype('float32'))
            Bs_gpu = pycuda.gpuarray.to_gpu(Bs.astype('float32'))

            # allocate arrays
            P_gpu = pycuda.gpuarray.empty((batch_size, num_factors), np.int32)
            info_gpu = pycuda.gpuarray.zeros(batch_size, np.int32)
            Cs_gpu = pycuda.gpuarray.empty_like(Bs_gpu) # inverted Bs.
            Rs_gpu = pycuda.gpuarray.empty_like(As_gpu) # final output, As * inverted Bs.
            
            # get pointer arrays
            A_arr = bptrs(As_gpu)
            B_arr = bptrs(Bs_gpu)
            C_arr = bptrs(Cs_gpu)
            R_arr = bptrs(Rs_gpu)

            allocated_shape[0] = As.shape
            allocations[0] = As_gpu, Bs_gpu, P_gpu, info_gpu, Cs_gpu, Rs_gpu, A_arr, B_arr, C_arr, R_arr


        handle = scikits.cuda.misc._global_cublas_handle

        # perform LU factorization
        cublas.cublasSgetrfBatched(handle, num_factors, B_arr.gpudata, num_factors, P_gpu.gpudata, info_gpu.gpudata, batch_size)
        # the LU factorization is now in Bs_gpu!

        # use factorization to perform inversion
        cublas.cublasSgetriBatched(handle, num_factors, B_arr.gpudata, num_factors, P_gpu.gpudata, C_arr.gpudata, num_factors, info_gpu.gpudata, batch_size)
        # the inverted matrices are now in Cs_gpu!

        # compute dot products dot(A, C) = dot(A, Binv). Note that the As are actually vectors!
        transb = 'n'
        transa = 'n'
        N, k, m = Cs_gpu.shape
        N2, l = As_gpu.shape
        n = 1 # As_gpu is a batch of vectors, not matrices, but we treat it as a batch of matrices with leading dimension 1.
        # kind of tricky, but it seems to work. The same goes for the output array Rs_gpu.

        lda = max(1, m)
        ldb = max(1, k)
        ldc = max(1, m)
        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        cublas.cublasSgemmBatched(handle, transb, transa, m, n, k, alpha, C_arr.gpudata,
                    lda, A_arr.gpudata, ldb, beta, R_arr.gpudata, ldc, N)

        # the resulting batch of vectors is now in Rs_gpu.
        return Rs_gpu.get()

##################################################################
##   helper functions                                           ##
##################################################################

def iter_rows(S):
    """
    Helper function to iterate quickly over the data and indices of the
    rows of the S matrix. A naive implementation using indexing
    on S is much, much slower.
    """
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        yield i, S.data[lo:hi], S.indices[lo:hi]