# Code belongs to the overview paper
#
# Generalized Normalizing Flows via Markov chains.
# P. Hagemann, J. Hertrich, G. Steidl (2021).
# Arxiv preprint arXiv:2111.12506
#
# The code is heavily based on our former paper Stochastic Normalizing Flows for Inverse problems
#
# Please cite the paper, if you find the code useful.
#
from torch.optim import Adam
import ot
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import time

from utils.Util_mixture import *
from utils.histogram_plot import make_image, make_image_multiple
from core.SNF import *
from core.INN import *

batch_size = 1024
num_samples_per_epoch = 1024
# train epochs where one epoch consists of one optimizer step
num_epochs = 5000

DIMENSION=100

reg=1e-10


def train_and_eval(mixture_params, b, convex_comb_factor, testing_ys, forward_map):
    # define forward model 
    forward_model=lambda x: forward_pass(x, forward_map)
    # define log posterior
    log_posterior=lambda samples,y:get_log_posterior(samples,forward_map,mixture_params,b,y)
    # create INN VAE MALA model
    inn_vae_mala = create_INN_VAE_MALA(4,128,log_posterior,metr_steps_per_block=3,dimension=DIMENSION,dimension_condition=DIMENSION,
                          num_inn_layers=1, step_size = 5e-5)
    print("INN VAE MALA has parameters:",sum(p.numel() for p in inn_vae_mala.parameters() if p.requires_grad))
    vae_mala = create_VAE_MALA(8,128,log_posterior,metr_steps_per_block=3,dimension=DIMENSION,dimension_condition=DIMENSION,
                          num_inn_layers=1,  step_size = 5e-5)
    print(" VAE MALA has parameters:",sum(p.numel() for p in vae_mala.parameters() if p.requires_grad))
    
    inn_mala = create_INN_MALA(8,128,log_posterior,metr_steps_per_block=3,dimension=DIMENSION,dimension_condition=DIMENSION,
                          num_inn_layers=1, step_size = 5e-5)
    print(" NF MALA has parameters:",sum(p.numel() for p in inn_mala.parameters() if p.requires_grad))
    inn_vae = create_VAE_INN(4,128,dimension=DIMENSION,dimension_condition=DIMENSION,
                          num_inn_layers=1)

    print(" NF VAE has parameters:",sum(p.numel() for p in inn_vae.parameters() if p.requires_grad))

    vae = create_VAE(8,128,dimension=DIMENSION,dimension_condition=DIMENSION)
    print(" VAE has parameters:",sum(p.numel() for p in vae.parameters() if p.requires_grad))

    INN = create_INN(8,128,dimension=DIMENSION,dimension_condition=DIMENSION)
    print(" NF has parameters:",sum(p.numel() for p in INN.parameters() if p.requires_grad))
    full_flow = create_full_flow(8,128,log_posterior,metr_steps_per_block=5,dimension=DIMENSION,dimension_condition=DIMENSION,
                              num_inn_layers=1, step_size = 5e-3, step_size2 = 5e-5)
    print(" fullflow has parameters:",sum(p.numel() for p in full_flow.parameters() if p.requires_grad))
    optimizer = Adam(inn_vae_mala.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer, inn_vae_mala, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()

    optimizer_vaemala = Adam(vae_mala.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer_vaemala, vae_mala, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()
    
    optimizer_inn_mala= Adam(inn_mala.parameters(), lr = 5e-4)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer_inn_mala, inn_mala, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()

    optimizer_innvae = Adam(inn_vae.parameters(), lr = 5e-4)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer_innvae, inn_vae, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()
    
    optimizer_vae = Adam(vae.parameters(), lr = 6e-4)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer_vae, vae, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr = 5e-4)
    prog_bar = tqdm(total=num_epochs)

    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_inn_epoch(optimizer_inn, INN, data_loader)
        prog_bar.set_description('INN Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()
    optimizer_ff = Adam(full_flow.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)

    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
        loss = train_SNF_epoch(optimizer_ff, full_flow, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params), convex_comb_factor=convex_comb_factor)
        prog_bar.set_description('Convex comb: {}, loss: {:.4f}, b: {}, n_mix: {}'.format(convex_comb_factor, loss, b, len(mixture_params)))
        prog_bar.update()
    prog_bar.close()
    testing_x_per_y = 1024

    testing_num_y = len(testing_ys)
    weights1, weights2 = np.ones((testing_x_per_y,)) / testing_x_per_y, np.ones((testing_x_per_y,)) / testing_x_per_y

    weights1 = weights1.astype(np.float64)
    weights2 = weights2.astype(np.float64)

    w1_sum = 0.
    w1=[]
    w2_sum = 0. 
    w2=[]
    w3_sum = 0.
    w3=[]
    w4_sum = 0.
    w4=[]
    w5_sum = 0.
    w5=[]
    w6_sum = 0. 
    w6=[]
    w7_sum = 0. 
    w7=[]
    
    tic=time.time()
    for i, y in enumerate(testing_ys):
        # get mixture model distribution of posterior
        true_posterior_params = get_mixture_posterior(mixture_params, forward_map, b**2, y)
        # draw samples
        true_posterior_samples = draw_mixture_dist(true_posterior_params, testing_x_per_y).cpu().numpy()
        inflated_ys = y[None, :].repeat(testing_x_per_y, 1)
        # latent samples for snf models
        inp_samps=torch.randn(testing_x_per_y, DIMENSION, device=device)
        
        samples_innvaemala = inn_vae_mala.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()
      
        samples_vae_mala = vae_mala.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()

        samples_inn_mala = inn_mala.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()
        
        samples_inn_vae = inn_vae.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()
        
        samples_vae = vae.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()

        samples_INN = INN(inp_samps, c = inflated_ys, rev = True)[0].detach().cpu().numpy()
        samples_ff = full_flow.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()

        make_image(true_posterior_samples, samples_innvaemala, 'INN_VAE_MALA{}.png'.format(i),directory='Images',inds=[0,49,99])
        make_image(true_posterior_samples, samples_INN, 'INN_{}.png'.format(i),directory='Images',inds=[0,49,99])
        make_image(true_posterior_samples, samples_vae_mala, 'VAE_MALA_{}.png'.format(i),directory='Images',inds=[0,49,99])

        make_image(true_posterior_samples, samples_inn_vae, 'INN_VAE_{}.png'.format(i),directory='Images',inds=[0,49,99])
        make_image(true_posterior_samples, samples_vae, 'VAE_{}.png'.format(i),directory='Images',inds=[0,49,99])
        make_image(true_posterior_samples, samples_inn_mala, 'INN_MALA{}.png'.format(i),directory='Images',inds=[0,49,99])
        make_image(true_posterior_samples, samples_ff, 'fullflow{}.png'.format(i),directory='Images',inds=[0,49,99])

        M_innvaemala =ot.dist(samples_innvaemala, true_posterior_samples, metric='euclidean')
        M_vaemala =ot.dist(samples_vae_mala, true_posterior_samples, metric='euclidean')
        M_innmala =ot.dist(samples_inn_mala, true_posterior_samples, metric='euclidean')
        M_innvae =ot.dist(samples_inn_vae, true_posterior_samples, metric='euclidean')
        M_vae =ot.dist(samples_vae, true_posterior_samples, metric='euclidean')
        M_inn =ot.dist(samples_INN, true_posterior_samples, metric='euclidean')
        M_ff =ot.dist(samples_ff, true_posterior_samples, metric='euclidean')

        w1.append(ot.emd2(weights1, weights2, M_innvaemala, numItermax=1000000))
        w2.append(ot.emd2(weights1, weights2, M_vaemala, numItermax=1000000))
        w3.append(ot.emd2(weights1, weights2, M_innmala, numItermax=1000000))
        w4.append(ot.emd2(weights1, weights2, M_innvae, numItermax=1000000))
        w5.append(ot.emd2(weights1, weights2, M_vae, numItermax=1000000))
        w6.append(ot.emd2(weights1, weights2, M_inn, numItermax=1000000))
        w7.append(ot.emd2(weights1, weights2, M_ff, numItermax=1000000))

        w1_sum+=w1[-1]
        w2_sum+=w2[-1]
        w3_sum+=w3[-1]
        w4_sum+=w4[-1]
        w5_sum+=w5[-1]

        w6_sum+=w6[-1]
        w7_sum+=w7[-1]


        toc=time.time()-tic
        print('Iteration: {} of {}, Time: {:.3f}, Time left (estimated): {:.3f}'.format(i+1,testing_num_y,toc,toc/(i+1)*(testing_num_y-i-1)))
        print('W_SNF: {:.3f},W_VAE_MALA: {:.3f},W_INN_MALA: {:.3f}, W_VAE_INN: {:.3f},W_VAE: {:.3f}, W_INN: {:.3f}, W_ff: {:.3f}'.format(w1[-1],w2[-1], w3[-1],w4[-1],w5[-1],w6[-1],w7[-1]))
    w1_mean=w1_sum / testing_num_y
    w2_mean=w2_sum / testing_num_y
    w3_mean=w3_sum / testing_num_y
    w4_mean=w4_sum / testing_num_y
    w5_mean=w5_sum / testing_num_y
    w6_mean=w6_sum / testing_num_y
    w7_mean=w7_sum / testing_num_y

    w1_std=np.std(w1)
    w2_std=np.std(w2)
    w3_std=np.std(w3)
    w4_std=np.std(w4)
    w5_std=np.std(w5)
    w6_std=np.std(w6)
    w7_std=np.std(w7)

    print('W SNF:', w1_mean,'+-',w1_std)
    print('W VAE_MALA:', w2_mean,'+-',w2_std)
    print('W_MALA:', w3_mean,'+-',w3_std)
    print('W INN_VAE:', w4_mean,'+-',w4_std)
    print('W VAE:', w5_mean,'+-',w5_std)

    print('W INN:', w6_mean,'+-',w6_std)
    print('W FF:', w7_mean,'+-',w7_std)



    return np.array((w1_mean,w2_mean,w3_mean,w4_mean,w5_mean,w6_mean,w7_mean))



testing_num_y = 100
b=0.05
forward_map = create_forward_model(scale = 0.1,dimension=DIMENSION)
n_mixtures=5

np.random.seed(0)
torch.manual_seed(0)
results_mean = np.zeros(7)
for rep in range(5):
    mixture_params=[]
    for i in range(n_mixtures):
        mixture_params.append((1./n_mixtures,torch.tensor(np.random.uniform(size=DIMENSION)*2-1, device = device,dtype=torch.float),torch.tensor(0.0001,device=device,dtype=torch.float)))
    testing_xs = draw_mixture_dist(mixture_params, testing_num_y)
    testing_ys = forward_pass(testing_xs, forward_map) + b * torch.randn(testing_num_y, DIMENSION, device=device)

    results_mean += train_and_eval(mixture_params,b,0.,testing_ys,forward_map)

print('FINAL MEANS')
print(results_mean/5)






