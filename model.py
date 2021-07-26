import torch as T
import torch.nn.functional as F
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

from argparse import ArgumentParser


class VAE(nn.Module):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.enc_out_dim = enc_out_dim
        self.input_height = input_height

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


    def encode(self, x):

        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        return mu, log_var
    
    def reparameterize(self, mu, std):
        
        q = T.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def decode(self, z):
        x_hat = self.decoder(z)

        return x_hat

    def forward(self, x):
        mu , log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), z, mu, log_var]

    def kl_divergence(self, z, mu, std):
        
        # --------------------------
        # Monte carlo estimation for computing KL divergence
        # --------------------------
        
        
        # First we define the prior, p, and variational distribution, q     
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # The we sample p(z) and q(z|x)
        log_qz_given_x = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl divergence
        kl = (log_qz_given_x - log_pz)
        kl = kl.sum(-1)
        
        return kl
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob. of seeing image under p(x|z)
        log_px_given_z = dist.log_prob(x)

        #Then, we sum over all the dimensions (color channels, x-axis, and y-axis)
        return log_px_given_z.sum(dim=(1, 2, 3))
    
    
    def loss_function(self, x):
    
        mu, log_var = self.encode(x)
        std = torch.exp(log_var / 2)

        z = self.reparameterize(mu, std)

        x_hat = self.decode(z)

        #reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        
        # kl divergence
        kl = self.kl_divergence(z, mu, std).mean()

        # nelbo
        nelbo = (kl - recon_loss)
        kl = kl.mean()
        recon_loss = recon_loss.mean()
        nelbo = nelbo.mean()

        summaries = dict([('loss', nelbo), ('elbo', -nelbo), ('kl', kl.mean()), ('rec_loss', recon_loss.mean())])

        return nelbo, summaries
    
    def sample(self, num_samples):
        p = torch.distributions.Normal(torch.zeros([latent_dim]), torch.ones([latent_dim]))
        z = p.rsample((num_samples,))
        samples = self.decode(z)

        return samples
    
    def generate_fake(self, x):
        return self.forward(x)[0]

