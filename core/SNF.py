import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, ActNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clamp=1.4

# negative log of gaussian energy
def gauss_energy(x):
    x=x.view(x.shape[0],-1)
    return 0.5*torch.sum(x**2, dim = 1)

# gradient of negative log of gaussian energy

def gauss_energy_grad(x):
    return x

def subnet_fc(c_in, c_out,sub_net_size,middle_layers=1):
    modules=[nn.Linear(c_in, sub_net_size), nn.ReLU()]
    for _ in range(middle_layers):
        modules.append(nn.Linear(sub_net_size, sub_net_size))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(sub_net_size,  c_out))
    return nn.Sequential(*modules)

# creates stochastic normalizing flow 

def create_snf(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,noise_std=0.4,num_inn_layers=1,
                 lang_steps = 0,lang_steps_prop=1, step_size = 5e-3,  langevin_prop = False):
    snf=SNF()
    for k in range(num_layers):
        lambd = (k+1)/(num_layers)
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
        if metr_steps_per_block>0:
            if lang_steps>0:
                snf.add_layer(Langevin_layer(log_posterior,lambd,lang_steps,step_size))
            if langevin_prop:
                snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
            else:
                snf.add_layer(MCMC_layer(log_posterior,lambd,noise_std,metr_steps_per_block))

    return snf

# creates INN VAE and MALA flow 
def create_INN_VAE_MALA(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,num_inn_layers=1,
                 lang_steps_prop=1, step_size = 5e-3):
    snf=SNF()
    for k in range(num_layers):
        lambd = (k+1)/num_layers
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
        snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size),
                                               backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size), noise_level = 1e-1))
    snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
    return snf

def create_full_flow(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,num_inn_layers=1,
                 lang_steps_prop=1, step_size = 5e-5, step_size2 = 5e-4):
    snf=SNF()
    for k in range(num_layers//2):
        lambd = (k+1)/num_layers
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
        snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
    for k in range(num_layers//2, num_layers):
        lambd = (k+1)/num_layers

        snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size),
                                                   backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size), noise_level = 1e-1))
    snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size2))
    return snf
# creates VAE and MALA flow 

def create_VAE_MALA(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,noise_std=0.4,num_inn_layers=1,
                 lang_steps = 0,lang_steps_prop=1, step_size = 5e-3, langevin_prop = False):
    snf=SNF()
    for k in range(num_layers):
        lambd = (k+1)/num_layers
        snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size),
                                               backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size), noise_level = 1e-1))
    snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
    return snf

# creates VAE flow

def create_VAE(num_layers, sub_net_size,dimension_condition=5,dimension=5):
    snf=SNF()
    for k in range(num_layers-1):
        lambd = (k+1)/num_layers        
        snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size),
                                                 backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size), noise_level = 1e-1))  
    snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size),
                                            backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = sub_net_size), noise_level = 1e-2))                                    
    return snf

# creates INN MALA flow

def create_INN_MALA(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,num_inn_layers=1,
                 lang_steps_prop=1, step_size = 5e-3):
    snf=SNF()
    for k in range(num_layers):
        lambd = (k+1)/num_layers
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))

    snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
    return snf

# creates VAE INN
def create_VAE_INN(num_layers, sub_net_size,num_inn_layers,dimension_condition=5,dimension=5):
    snf=SNF()
    for k in range(num_layers-1):
        #lambd = (k+1)/num_layers
        lambd = 1.
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
        snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = 128),
                                                backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = 128), noise_level = 1e-1))
    snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
    snf.add_layer(learned_stochastic_layer(forward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = 128),
                                            backward_constructor = lambda: subnet_fc(dimension+dimension_condition, dimension, sub_net_size = 128), noise_level = 1e-2))
    return snf


# creates SNF class
# every MCMC, Langevin, deterministic INN or VAE is added as a layer here
class SNF(nn.Module):
    def __init__(self,layers=[]):
        super(SNF, self).__init__()
        self.layers=[]
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self,layer):
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)),layer)
    
    def forward(self,zs,ys=None):
        logdet = torch.zeros(len(zs), device = device)
        for k in range(len(self.layers)):
            out = self.layers[k].forward(zs, ys)
            zs = out[0]
            logdet += out[1]
            #print(k,logdet.mean().item())
        return (zs,logdet)
    
    def forward_path(self,zs,ys=None):
        path = []
        path.append(zs)
        for k in range(len(self.layers)):
            out = self.layers[k].forward(zs, ys)
            zs = out[0]
            path.append(out[0])
        return (path)

    def backward(self,zs, ys=None):
        logdet = torch.zeros(len(zs), device = device)
        for k in range(len(self.layers)-1,-1, -1):
            out = self.layers[k].backward(zs, ys)
            zs = out[0]
            logdet += out[1]
        return (zs,logdet)

# creates VAE class (learned stochastic layer)
# note that here we only learn the means of encoder/decoder networks with fixed diagonal Covariance

class learned_stochastic_layer(nn.Module):
    def __init__(self,forward_constructor,backward_constructor,noise_level=1e-2):
        super(learned_stochastic_layer, self).__init__()
        self.forward_model=forward_constructor().to(device)
        self.backward_model=backward_constructor().to(device)
        self.add_module("forward_model",self.forward_model)
        self.add_module("backward_model",self.backward_model)
        self.forward_noise_level=torch.tensor(noise_level,device=device,dtype=torch.float)
        self.backward_noise_level=torch.tensor(noise_level,device=device,dtype=torch.float)

    def forward(self,xs,c=None):
        xs_forward=self.forward_model(torch.cat((xs,c),1))
        forward_noise=torch.randn_like(xs_forward)
        xs_new=xs_forward+self.forward_noise_level*forward_noise
        xs_new_backward=self.backward_model(torch.cat((xs_new,c),1))
        backward_noise=(xs_new_backward-xs)/self.backward_noise_level
        logdet= 0.5 * (-(backward_noise**2).view(xs.shape[0],-1).sum(axis=1))
        return xs_new,logdet
        
    def backward(self,xs,c=None):
        xs_net=self.backward_model(torch.cat((xs,c),1))
        forward_noise=torch.randn_like(xs_net)
        xs_new=xs_net+self.backward_noise_level*forward_noise
        xs_new_net=self.forward_model(torch.cat((xs_new,c),1))
        backward_noise=(xs_new_net-xs)/self.forward_noise_level
        logdet= 0.5 * (-(backward_noise**2).view(xs.shape[0],-1).sum(axis=1))

        return xs_new,logdet

# deterministic INN layer

class deterministic_layer(nn.Module):
    def __init__(self,num_inn_layers,sub_net_size,dimension_condition=5,dimension=5):
        super(deterministic_layer, self).__init__()
        k=1
        self.conditional=dimension_condition is not None
        nodes = [InputNode(dimension, name=F'input_{k}')]
        cond = ConditionNode(dimension_condition, name=F'condition_{k}')

        for i in range(num_inn_layers):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                                  {'subnet_constructor':lambda c_in,c_out: subnet_fc(c_in,c_out,sub_net_size), 'clamp':clamp},
                                  conditions = cond,
                                  name=F'coupling_{k}'))
        nodes.append(OutputNode(nodes[-1], name=F'output_{k}'))
        self.model=ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
        self.add_module("model",self.model)

    def forward(self,xs,c=None):
        return self.model(xs,c=c, rev = True)

    def backward(self,xs,c=None):
        return self.model(xs,c=c)


# creates MCMC layer with log posterior to anneal to.
# proposal density is with a gaussian noise
# forward layer is backward layer
class MCMC_layer(nn.Module):
    def __init__(self,log_posterior,lambd,noise_std,metr_steps_per_block):
        super(MCMC_layer,self).__init__()
        self.noise_std=noise_std
        self.metr_steps_per_block=metr_steps_per_block
        self.get_log_posterior=log_posterior
        self.lambd=lambd
    def forward(self,xs,c=None):
        zs,e=anneal_to_energy(xs, get_interpolated_energy_fun(c, self.lambd,self.get_log_posterior),self.metr_steps_per_block,noise_std=self.noise_std)
        e=e.view(len(e))
        return zs,e
    def backward(self,xs,c=None):
        return self.forward(xs,c=c)
    
# creates MALA layer, i.e. MCMC layer with a langevin proposal

class MALA_layer(nn.Module):
    def __init__(self,log_posterior,lambd,metr_steps_per_block,lang_steps,stepsize):
        super(MALA_layer,self).__init__()
        self.metr_steps_per_block=metr_steps_per_block
        self.get_log_posterior=log_posterior
        self.lambd=lambd
        self.lang_steps=lang_steps
        self.stepsize=stepsize
    def forward(self,xs,c=None):
        zs,e=anneal_to_energy(xs, get_interpolated_energy_fun(c, self.lambd,self.get_log_posterior),self.metr_steps_per_block,langevin_prop=True,lang_steps=self.lang_steps,stepsize=self.stepsize)
        e=e.view(len(e))
        return zs,e
    def backward(self,xs,c=None):
        return self.forward(xs,c=c)

# creates langevin layer without accept/reject step

class Langevin_layer(nn.Module):
    def __init__(self,log_posterior,lambd,lang_steps,stepsize,z_energy=lambda x,y: gauss_energy(x),damp_beta=1.):
        super(Langevin_layer,self).__init__()
        self.get_log_posterior=log_posterior
        self.lambd=lambd
        self.lang_steps=lang_steps
        self.stepsize=stepsize
        self.z_energy=z_energy
        self.damp_beta=damp_beta
    def forward(self,xs,c=None):
        zs,dW,_,_=langevin_step(xs,self.stepsize,get_interpolated_energy_fun(c, self.lambd,self.get_log_posterior),self.lang_steps,damp_beta=self.damp_beta)
        return zs,dW
    def backward(self,xs,c=None):
        return self.forward(xs,c=c)
    
# here the proposal/intermediate densities are defined as a geometric mean of the latent and target (posterior) density
# anneals linaerly in log density space

def get_interpolated_energy_fun(ys,lambd,get_log_posterior):
    if lambd==0.:
        def energy(x):
            return 0.5*torch.sum(x**2, dim = 1)
        return energy
    if lambd==1.:
        def energy(x):
            return get_log_posterior(x, ys)
        return energy
    def energy(x):
        return lambd*(get_log_posterior(x, ys)).view(len(x))+(1-lambd)*0.5*torch.sum(x**2, dim = 1)
    return energy

# gradients of energy for langevin methods

def energy_grad(x, energy):
    x = x.requires_grad_(True)
    e = energy(x)
    return torch.autograd.grad(e.sum(), x,retain_graph=True)[0],e


# anneals to energy with accept/reject schemes like in MCMC

def anneal_to_energy(x_curr, energy,metr_steps_per_block,noise_std=0.1, langevin_prop=False, lang_steps=None, stepsize=None):
    e0 = energy(x_curr)
    for i in range(metr_steps_per_block):
        if langevin_prop == True:
            x_prop, work_diff,e_curr,e_prop = langevin_step(x_curr, stepsize, energy, lang_steps)
            prob_diff = torch.exp(-e_prop+e_curr+work_diff)
        else:
            noise = noise_std * torch.randn_like(x_curr, device = device)
            x_prop = x_curr + noise
            e_prop=energy(x_prop)
            e_curr=energy(x_curr)
            prob_diff = torch.exp(-e_prop+e_curr)
        r = torch.rand_like(prob_diff, device = device)
        acc = (r < prob_diff).float().view(len(x_prop),1)
        rej = 1. - acc
        rej = rej.view(len(rej),1)
        x_curr = rej * x_curr + acc * x_prop
    e = rej * e_curr.view(len(e_curr),1) + acc * e_prop.view(len(e_prop),1)
    return (x_curr, e.view(len(e))-e0.view(len(e)))

# performs a langevin steps

def langevin_step(x, stepsize, energy, lang_steps,damp_beta=1.):
        logdet = torch.zeros((x.shape[0], 1), device = device)
        # damping coefficient. setting it larger than 1 reduces noise but changes stationary distribution!
        beta=damp_beta
        for i in range(lang_steps):
            # forward noise
            eta = torch.randn_like(x, device = device)
            # forward step
            grad_x,e_x=energy_grad(x,energy)
            if i==0:
                energy_x=e_x
            y = x - stepsize * grad_x + np.sqrt(2*stepsize/beta) * eta
            # backward noise
            grad_y,energy_y=energy_grad(y, energy)
            eta_ = (x - y + stepsize* grad_y) / np.sqrt(2*stepsize/beta)
            # noise ratio
            logdet += 0.5 * (eta**2 - eta_**2).sum(axis=1, keepdims=True)
            # update state
            x = y
        return (x, logdet.view(len(x)),energy_x,energy_y)
    
# trains SNF with forward KL loss

def train_SNF_epoch(optimizer, snf, epoch_data_loader,forward_model,a,b,get_prior_log_likelihood, convex_comb_factor=0.5):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        loss = 0

        out = snf.backward(x, y)
        invs = out[0]
        jac_inv = out[1]
        l5 = 0.5* torch.sum(invs**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size * (1-convex_comb_factor))
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean_loss




