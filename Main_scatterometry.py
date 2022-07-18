# This code belongs to the paper
#
# Generalized Normalizing Flows via Markov chains.
# P. Hagemann, J. Hertrich, G. Steidl (2021).
# Arxiv preprint arXiv:2111.12506
#
# Please cite the paper, if you use this code.
# This script reproduces the numerical example from Section 7.2 of the paper.
#
from torch.optim import Adam
import torch
import numpy as np
from tqdm import tqdm
import scipy
import time
import os

from utils.Util_scatterometry import *
from utils.histogram_plot import make_image, make_image_multiple
from core.SNF import *
from core.INN import *

# define parameters
num_epochs = 5000
batch_size = 1600
DIMENSION = 3




# MCMC parameters for discovering "ground truth"
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
lambd_bd = 100
a = 0.2
b = 0.01

relu = torch.nn.ReLU()
# regularization parameter for KL calculation
reg = 1e-10



# load forward model
forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  23)).to(device)

forward_model.load_state_dict(torch.load('models_scatterometry/forward_model_new.pt'))
for param in forward_model.parameters():
    param.requires_grad=False


def train_and_eval(a, b, testing_ys, forward_model):
    # log posterior defined
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd)
    # create models
    snf = create_INN_MALA(4,64,log_posterior,metr_steps_per_block=3,dimension=3,dimension_condition=23,
                          num_inn_layers=1, step_size = 1e-3)
    print("NF MALA has parameters:",sum(p.numel() for p in snf.parameters() if p.requires_grad))

    INN = create_INN(4,64,dimension=3,dimension_condition=23)
    print("NF has parameters:",sum(p.numel() for p in INN.parameters() if p.requires_grad))

    vae_mala = create_VAE_MALA(4,64,log_posterior,metr_steps_per_block=3,dimension=3,dimension_condition=23,
                          num_inn_layers=1, step_size = 1e-3)
    print("VAE MALA has parameters:",sum(p.numel() for p in vae_mala.parameters() if p.requires_grad))
    vae = create_VAE(4,64,dimension=3,dimension_condition=23)
     
    print("vae has parameters:",sum(p.numel() for p in vae.parameters() if p.requires_grad))
    # train models 
    fullflow = create_full_flow(4,64,log_posterior,metr_steps_per_block=5,dimension=3,dimension_condition=23,
                              num_inn_layers=1, step_size = 1e-2, step_size2 = 1e-3)

    print("full flow has parameters:",sum(p.numel() for p in fullflow.parameters() if p.requires_grad))

    optimizer = Adam(snf.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer, snf, data_loader,forward_model, a, b,None,convex_comb_factor=0.)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr = 1e-3)


    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_inn_epoch(optimizer_inn, INN, data_loader)
        prog_bar.set_description('determ INN loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_vae = Adam(vae_mala.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer_vae, vae_mala, data_loader,forward_model, a, b,None,convex_comb_factor=0.)
        prog_bar.set_description('VAE MALA loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_auto = Adam(vae.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer_auto, vae, data_loader,forward_model, a, b,None,convex_comb_factor=0.)
        prog_bar.set_description('VAE loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    
    optimizer_ff = Adam(fullflow.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer_ff, fullflow, data_loader,forward_model, a, b,None,convex_comb_factor=0.)
        prog_bar.set_description('FF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    
    # parameters for testing
    testing_x_per_y = 16000

    testing_num_y = len(testing_ys)

    kl1_sum = 0.
    kl2_sum = 0.
    kl3_sum = 0.
    kl4_sum = 0.
    kl5_sum = 0.

    kl1_vals=[]
    kl2_vals=[]
    kl3_vals=[]
    kl4_vals=[]
    kl5_vals=[]

    nbins=50
    repeats=10
    tic=time.time()
    for i, y in enumerate(testing_ys):
        # testing
        hist_mcmc_sum=np.zeros((nbins,nbins,nbins))
        hist_snf_sum=np.zeros((nbins,nbins,nbins))
        hist_inn_sum=np.zeros((nbins,nbins,nbins))
        hist_vae_mala_sum=np.zeros((nbins,nbins,nbins))
        hist_vae_sum=np.zeros((nbins,nbins,nbins))
        hist_ff_sum=np.zeros((nbins,nbins,nbins))

        for asdf in range(repeats):
            # run methods for posterior samplings
            inflated_ys = y[None, :].repeat(testing_x_per_y, 1)
            mcmc_energy=lambda x:get_log_posterior(x,forward_model,a,b,inflated_ys,lambd_bd)
            # generate ground truth posterior distribution via MCMC
            true_posterior_samples = anneal_to_energy(torch.rand(testing_x_per_y,3, device = device)*2-1,mcmc_energy,METR_STEPS,noise_std=NOISE_STD_MCMC )[0].detach().cpu().numpy()
            # generate samples 
            samples1 = snf.forward(torch.randn(testing_x_per_y, DIMENSION, device=device), inflated_ys)[0].detach().cpu().numpy()
            samples2 = INN(torch.randn(testing_x_per_y, DIMENSION, device=device), c = inflated_ys, rev = True)[0].detach().cpu().numpy()
            samples3 = vae_mala(torch.randn(testing_x_per_y, DIMENSION, device=device), inflated_ys)[0].detach().cpu().numpy()
            samples4 = vae(torch.randn(testing_x_per_y, DIMENSION, device=device), inflated_ys)[0].detach().cpu().numpy()
            samples5 = fullflow(torch.randn(testing_x_per_y, DIMENSION, device=device), inflated_ys)[0].detach().cpu().numpy()

            # generate histograms for KL evaluations
            hist_mcmc,_ = np.histogramdd(true_posterior_samples, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_snf,_ = np.histogramdd(samples1, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_inn,_ = np.histogramdd(samples2, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_vae_mala,_ = np.histogramdd(samples3, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_vae,_ = np.histogramdd(samples4, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_ff,_ = np.histogramdd(samples5, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))

            hist_mcmc_sum+=hist_mcmc
            hist_snf_sum+=hist_snf
            hist_inn_sum+=hist_inn
            hist_vae_mala_sum+=hist_vae_mala
            hist_vae_sum+=hist_vae
            hist_ff_sum+=hist_ff


        # save histograms
        make_image(true_posterior_samples,samples1, 'SNF_img'+str(i),'scatterometry_images')
        make_image(true_posterior_samples,samples2, 'INN_img'+str(i),'scatterometry_images')
        make_image(true_posterior_samples,samples3, 'VAE_MALA_img'+str(i),'scatterometry_images')
        make_image(true_posterior_samples,samples4, 'VAE_img'+str(i),'scatterometry_images')
        make_image(true_posterior_samples,samples5, 'FF_img'+str(i),'scatterometry_images')

        hist_mcmc = hist_mcmc_sum/hist_mcmc_sum.sum()
        hist_snf = hist_snf_sum/hist_snf_sum.sum()
        hist_inn = hist_inn_sum/hist_inn_sum.sum()
        hist_vae_mala = hist_vae_mala_sum/hist_vae_mala_sum.sum()
        hist_vae = hist_vae_sum/hist_vae_sum.sum()
        hist_ff = hist_ff_sum/hist_ff_sum.sum()

        hist_mcmc+=reg
        hist_snf+=reg
        hist_inn+=reg
        hist_vae_mala+=reg
        hist_vae+=reg
        hist_ff+=reg

        hist_mcmc/=hist_mcmc.sum()
        hist_snf/=hist_snf.sum()
        hist_inn/=hist_inn.sum()
        hist_vae_mala/=hist_vae_mala.sum()
        hist_vae/=hist_vae.sum()
        hist_ff/=hist_ff.sum()

        # evaluate KL divergence
        kl1=np.sum(scipy.special.rel_entr(hist_mcmc,hist_snf))
        kl2=np.sum(scipy.special.rel_entr(hist_mcmc,hist_inn))
        kl3=np.sum(scipy.special.rel_entr(hist_mcmc,hist_vae_mala))
        kl4=np.sum(scipy.special.rel_entr(hist_mcmc,hist_vae))
        kl5=np.sum(scipy.special.rel_entr(hist_mcmc,hist_ff))

        kl1_sum += kl1
        kl2_sum += kl2
        kl3_sum += kl3
        kl4_sum += kl4
        kl5_sum += kl5

        kl1_vals.append(kl1)
        kl2_vals.append(kl2)
        kl3_vals.append(kl3)
        kl4_vals.append(kl4)
        kl5_vals.append(kl5)

        toc=time.time()-tic
        print('Iteration: {} of {}, Time: {:.3f}, Time left (estimated): {:.3f}'.format(i+1,testing_num_y,toc,toc/(i+1)*(testing_num_y-i-1)))
        print('KL_SNF: {:.3f}, KL_INN {:.3f},, KL_VAE_MALA {:.3f}, KL_VAE {:.3f}, KL_FF{:.3f}'.format(kl1,kl2,kl3,kl4,kl5))
    kl1_vals=np.array(kl1_vals)
    kl2_vals=np.array(kl2_vals)
    kl3_vals = np.array(kl3_vals)
    kl4_vals = np.array(kl4_vals)
    kl5_vals = np.array(kl5_vals)

    kl1_var=np.sum((kl1_vals-kl1_sum/testing_num_y)**2)/testing_num_y
    kl2_var=np.sum((kl2_vals-kl2_sum/testing_num_y)**2)/testing_num_y
    kl3_var=np.sum((kl3_vals-kl3_sum/testing_num_y)**2)/testing_num_y
    kl4_var=np.sum((kl4_vals-kl4_sum/testing_num_y)**2)/testing_num_y
    kl5_var=np.sum((kl5_vals-kl5_sum/testing_num_y)**2)/testing_num_y

    print('KL1:', kl1_sum / testing_num_y,'+-',kl1_var)
    print('KL2:', kl2_sum / testing_num_y,'+-',kl2_var)
    print('KL3:', kl3_sum / testing_num_y,'+-',kl3_var)
    print('KL4:', kl4_sum / testing_num_y,'+-',kl4_var)
    print('KL5:', kl5_sum / testing_num_y,'+-',kl5_var)

    return np.array((kl1_sum / testing_num_y, kl2_sum/testing_num_y, kl3_sum / testing_num_y, kl4_sum/testing_num_y,kl5_sum / testing_num_y))

# define seed and call methods
torch.manual_seed(0)
np.random.seed(0)
results_mean = np.zeros(5)
for rep in range(3):
    # number of testing_ys
    testing_num_y = 20
    # prior distribution is uniform
    testing_xs = torch.rand(testing_num_y, 3, device = device)*2-1
    print(testing_xs)
    # testing_ys creation using error model
    testing_ys = forward_model(testing_xs) + b * torch.randn_like(forward_model(testing_xs)) + forward_model(testing_xs)*a*torch.randn_like(forward_model(testing_xs))

    # evaluate KL distances
    results_mean += train_and_eval(a, b, testing_ys, forward_model)

print('Results mean:')
print(results_mean/3)









