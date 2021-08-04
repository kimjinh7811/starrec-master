import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class RecVAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(RecVAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        [self.hidden_dim, self.latent_dim] = model_conf['hidden_latent']
        self.beta = model_conf['beta']
        self.dropout = model_conf['dropout']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        
        self.lr = model_conf['lr']
        self.lr_decay = model_conf['lr_decay']
        self.lr_decay_from = model_conf['lr_decay_from']
        self.lr_decay_step = model_conf['lr_decay_step']
        self.lr_decay_rate = model_conf['lr_decay_rate']

        self.num_opt_dec_per_epoch = model_conf['num_opt_dec_per_epoch']
        self.num_opt_enc_per_epoch = self.num_opt_dec_per_epoch * 3

        self.device=device
        self.build_graph()

    def build_graph(self):
        self.fc1 = nn.Linear(self.num_items, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln5 = nn.LayerNorm(self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)

        self.prior = GaussianMixturePriorWithAprPost(self.latent_dim, self.num_users)
        self.decoder = DeterministicDecoder(self.latent_dim, self.num_items)
        
        # optimizer
        decoder_params = set(self.decoder.parameters())
        embedding_params = set(self.prior.user_mu.parameters()) | set(self.prior.user_logvar.parameters())
        encoder_params = set(self.parameters()) - decoder_params - embedding_params

        self.encoder_opt = torch.optim.Adam(encoder_params, lr=self.lr)
        self.decoder_opt = torch.optim.Adam(decoder_params, lr=self.lr)
        self.embedding_opt = torch.optim.Adam(embedding_params, lr=self.lr)

        if self.lr_decay:
            self.enc_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_opt, self.lr_decay_step, self.lr_decay_rate)
            self.dec_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_opt, self.lr_decay_step, self.lr_decay_rate)
            self.emb_scheduler = torch.optim.lr_scheduler.StepLR(self.embedding_opt, self.lr_decay_step, self.lr_decay_rate)
        else:
            self.enc_scheduler, self.dec_scheduler, self.emb_scheduler = None, None, None

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, user_ratings, user_idx, beta=1, dropout_rate=0.5, calculate_loss=True, mode=None):
        if mode == 'pr':
            mu, logvar = self.encode(user_ratings, dropout_rate=dropout_rate)
        elif mode == 'mf':
            mu, logvar = self.encode(user_ratings, dropout_rate=0)
            
        z = self.reparameterize(mu, logvar)
        x_pred, decoder_loss = self.decode(z)
        
        NLL = -(F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
        
        
        if calculate_loss:
            if mode == 'pr':
                norm = user_ratings.sum(dim=-1)
                KLD = -(self.prior(z, user_idx) - log_norm_pdf(z, mu, logvar)).sum(dim=-1).mul(norm).mean()
                loss = NLL + beta * KLD + decoder_loss
            
            elif mode == 'mf':
                KLD = NLL * 0
                loss = NLL + decoder_loss
            
            return (NLL, KLD), loss
            
        else:
            return x_pred

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix
        train_users = train_matrix.shape[0]
        train_data_perm = np.arange(train_users)

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):            
            self.train()
            epoch_train_start = time()
            # ===================== optimize encoder
            # lr decay after warm-up epoch
            if self.lr_decay and epoch >= self.lr_decay_from:
                self.enc_scheduler.step()
                self.dec_scheduler.step()

            enc_nll, enc_kld = self.optimize_half(train_matrix, is_encoder=True, beta=self.beta, mode='pr')            
            # ===================== update prior 
            self.set_embeddings(train_matrix)            
            # ===================== optimize decoder
            dec_nll, dec_kld = self.optimize_half(train_matrix, is_encoder=False, beta=None, mode='mf')
            
            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch, \
                'enc_nll: %.3f' % enc_nll, 'enc_kld: %.3f' % enc_kld,
                'dec_nll: %.3f' % dec_nll, 'dec_kld: %.3f' % dec_kld,
                'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()
                
                valid_score = evaluator.evaluate(self)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))
        
        total_train_time = time() - start
        return early_stop.best_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_matrix = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_matrix, user_ids, calculate_loss=False, mode='mf').detach().cpu().numpy()
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
            return eval_output

    def optimize_half(self, input_matrix, is_encoder=True, beta=None, mode='pr', logger=None):
        if is_encoder:
            num_opt = self.num_opt_enc_per_epoch
            optimizer = self.encoder_opt
        else:
            num_opt = self.num_opt_dec_per_epoch
            optimizer = self.decoder_opt

        users = np.arange(input_matrix.shape[0])
        NLL_total = []
        KLD_total = []
        for epoch in range(num_opt):
            NLL_loss = 0
            KLD_loss = 0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            for user_idx in batch_loader:
                optimizer.zero_grad()
                batch_matrix = torch.FloatTensor(input_matrix[user_idx].toarray()).to(self.device)
                user_idx = torch.LongTensor(user_idx).to(self.device)

                (NLL, KLD), loss = self.forward(batch_matrix, user_idx, beta=beta, mode=mode)
                loss.backward()

                optimizer.step()

                NLL_loss += NLL.item()
                KLD_loss += KLD.item()
            
            # print('NLL_loss:', NLL_loss, 'KLD_loss:', KLD_loss)
            
            NLL_total.append(NLL_loss)
            KLD_total.append(KLD_loss)
        return np.mean(NLL_total), np.mean(KLD_total)

    def set_embeddings(self, input_matrix, momentum=0, weight=None):
        istraining = self.training
        self.eval()
        
        users = np.arange(input_matrix.shape[0])
        batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
        for batch_idx in batch_loader:
            batch_matrix = torch.FloatTensor(input_matrix[batch_idx].toarray()).to(self.device)

            new_user_mu, new_user_logvar = self.encode(batch_matrix, 0)

            old_user_mu = self.prior.user_mu.weight.data[batch_idx,:].detach()
            old_user_logvar = self.prior.user_logvar.weight.data[batch_idx,:].detach()

            if weight:
                old_user_var = torch.exp(old_user_logvar)
                new_user_var = torch.exp(new_user_logvar)

                post_user_var = 1 / (1 / old_user_var + weight / new_user_var)
                post_user_mu = (old_user_mu / old_user_var + weight * new_user_mu / new_user_var) * post_user_var

                self.prior.user_mu.weight.data[batch_idx,:] = post_user_mu
                self.prior.user_logvar.weight.data[batch_idx,:] = torch.log(post_user_var + new_user_var)
            else:
                self.prior.user_mu.weight.data[batch_idx,:] = momentum * old_user_mu + (1-momentum) * new_user_mu
                self.prior.user_logvar.weight.data[batch_idx,:] = momentum * old_user_logvar + (1-momentum) * new_user_logvar

        if istraining:
            self.train()
        else:
            self.eval()

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def encode(self, x, dropout_rate=0.8):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        
        return self.fc21(h5), self.fc22(h5)    
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

class DeterministicDecoder(nn.Linear):
    def __init__(self, *args):
        super(DeterministicDecoder, self).__init__(*args)

    def forward(self, *args):
        output = super(DeterministicDecoder, self).forward(*args)
        return output, 0


class GaussianMixturePriorWithAprPost(nn.Module):
    def __init__(self, latent_dim, input_count):
        super(GaussianMixturePriorWithAprPost, self).__init__()
        
        self.gaussians_number = 1
        
        self.mu_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_uniform_prior.data.fill_(10)
        
        self.user_mu = nn.Embedding(input_count, latent_dim)
        self.user_logvar = nn.Embedding(input_count, latent_dim)
        
    def forward(self, z, idx):
        density_per_gaussian1 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, :, :].detach(),
                                            logvar=self.logvar_prior[None, :, :].detach()
                                           ).add(np.log(1/5 - 1/20))
        
        
        density_per_gaussian2 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.user_mu(idx)[:, :, None].detach(),
                                            logvar=self.user_logvar(idx)[:, :, None].detach()
                                           ).add(np.log(4/5 - 1/20))
        
        density_per_gaussian3 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, :, :].detach(),
                                            logvar=self.logvar_uniform_prior[None, :, :].detach()
                                           ).add(np.log(1/10))
        
        density_per_gaussian = torch.cat([density_per_gaussian1,
                                          density_per_gaussian2,
                                          density_per_gaussian3], dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

def swish_(x):
    return x.mul_(torch.sigmoid(x))

def swish(x):
    return x.mul(torch.sigmoid(x))

def kl(q_distr, p_distr, weights, eps=1e-7):
    mu_q, logvar_q = q_distr
    mu_p, logvar_p = p_distr
    return 0.5 * (((logvar_q.exp() + (mu_q - mu_p).pow(2)) / (logvar_p.exp() + eps)                     + logvar_p - logvar_q - 1
                   ).sum(dim=-1) * weights).mean()

def simple_kl(mu_q, logvar_q, logvar_p_scale, norm):
    return (-0.5 * ( (1 + logvar_q #- torch.log(torch.ones(1)*logvar_p_scale) \
                      - mu_q.pow(2)/logvar_p_scale - logvar_q.exp()/logvar_p_scale
                     )
                   ).sum(dim=-1) * norm
           ).mean()

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

def log_norm_std_pdf(x):
    return -0.5*(np.log(2 * np.pi) + x.pow(2))