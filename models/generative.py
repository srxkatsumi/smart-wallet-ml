"""
Família: Modelos Generativos
Modelos: VAE (Variational Autoencoder), GAN (Generative Adversarial Network)

VAE (Kingma & Welling, 2014 — "Auto-Encoding Variational Bayes", ICLR 2014):
  Aprende uma representação latente contínua dos dados via ELBO
  (Evidence Lower BOund). O encoder mapeia x → (mu, logvar), o decoder
  mapeia z → x_rec. Após o treino, um classificador linear é ajustado
  sobre as representações latentes mu — separação linear no espaço latente
  indica estrutura aprendida.

GAN — Discriminador como Estimador de Densidade (Goodfellow et al., 2014 +
  Odena et al., 2017 ACGAN):
  Os exemplos y=1 são tratados como "reais" e y=0 como "gerados". O
  discriminador aprende D(x) = P(x vem da distribuição UP) ≈ P(y=1|x).
  O gerador serve de regularizador implícito do discriminador. Esta
  interpretação como estimador de razão de densidades está fundamentada
  em Mohamed & Lakshminarayanan, 2016 ("Learning in Implicit Generative
  Models").

Interface (compatível com ensemble.py):
  train(X, y)            -> model_dict
  predict(model_dict, X) -> np.ndarray de probabilidades em [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

LATENT_DIM  = 16
HIDDEN      = 64
EPOCHS_VAE  = 50
EPOCHS_GAN  = 30
LR          = 1e-3
BATCH_SIZE  = 64
NOISE_DIM   = 32


# ── VAE ───────────────────────────────────────────────────────────────────

class _VAE(nn.Module):
    def __init__(self, input_size: int, hidden: int, latent_dim: int):
        super().__init__()
        self.enc_fc   = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU())
        self.enc_mu   = nn.Linear(hidden, latent_dim)
        self.enc_logv = nn.Linear(hidden, latent_dim)
        self.dec      = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def encode(self, x):
        h = self.enc_fc(x)
        return self.enc_mu(h), self.enc_logv(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z    = self.reparameterize(mu, logvar)
        xrec = self.dec(z)
        return xrec, mu, logvar


def _vae_loss(xrec, x, mu, logvar):
    recon = nn.functional.mse_loss(xrec, x, reduction="sum")
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / x.size(0)


def _train_vae(X_sc: np.ndarray, y: np.ndarray) -> tuple:
    n_feat = X_sc.shape[1]
    vae    = _VAE(n_feat, HIDDEN, LATENT_DIM)
    opt    = torch.optim.Adam(vae.parameters(), lr=LR)
    X_t    = torch.from_numpy(X_sc.astype(np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    vae.train()
    for _ in range(EPOCHS_VAE):
        for (xb,) in loader:
            opt.zero_grad()
            xrec, mu, logvar = vae(xb)
            _vae_loss(xrec, xb, mu, logvar).backward()
            opt.step()

    vae.eval()
    with torch.no_grad():
        mu_all, _ = vae.encode(X_t)
    Z = mu_all.numpy()

    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(Z, y.astype(int))
    return vae, clf


def _predict_vae(vae: _VAE, clf: LogisticRegression,
                 X_sc: np.ndarray) -> np.ndarray:
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(torch.from_numpy(X_sc.astype(np.float32)))
    return clf.predict_proba(mu.numpy())[:, 1]


# ── GAN ───────────────────────────────────────────────────────────────────

class _Generator(nn.Module):
    def __init__(self, noise_dim: int, output_size: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),   nn.LeakyReLU(0.2),
            nn.Linear(hidden, output_size),
        )

    def forward(self, z):
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),   nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),  nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _train_gan(X_sc: np.ndarray, y: np.ndarray) -> _Discriminator:
    n_feat = X_sc.shape[1]
    G      = _Generator(NOISE_DIM, n_feat, HIDDEN)
    D      = _Discriminator(n_feat, HIDDEN)
    opt_G  = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D  = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    # "real" = y==1 samples, "fake" gerados pelo G
    real_idx = np.where(y == 1)[0]
    X_real   = torch.from_numpy(X_sc[real_idx].astype(np.float32))

    for _ in range(EPOCHS_GAN):
        # ─ treinar D ─
        opt_D.zero_grad()
        real_labels = torch.ones(len(X_real), 1)
        loss_real   = loss_fn(D(X_real), real_labels)

        z     = torch.randn(len(X_real), NOISE_DIM)
        fake  = G(z).detach()
        fake_labels = torch.zeros(len(X_real), 1)
        loss_fake   = loss_fn(D(fake), fake_labels)
        (loss_real + loss_fake).backward()
        opt_D.step()

        # ─ treinar G ─
        opt_G.zero_grad()
        z    = torch.randn(len(X_real), NOISE_DIM)
        fake = G(z)
        loss_fn(D(fake), torch.ones(len(X_real), 1)).backward()
        opt_G.step()

    return D


def _predict_gan(D: _Discriminator, X_sc: np.ndarray) -> np.ndarray:
    D.eval()
    with torch.no_grad():
        probs = D(torch.from_numpy(X_sc.astype(np.float32))).squeeze(1).numpy()
    return probs


# ── Interface unificada ───────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = RobustScaler()
    scaler.fit(X)
    X_sc   = scaler.transform(X)
    vae, clf = _train_vae(X_sc, y)
    D        = _train_gan(X_sc, y)
    return {"scaler": scaler, "vae": vae, "clf": clf, "D": D}


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    X_sc  = model_dict["scaler"].transform(X)
    p_vae = _predict_vae(model_dict["vae"], model_dict["clf"], X_sc)
    p_gan = _predict_gan(model_dict["D"], X_sc)
    return (p_vae + p_gan) / 2.0
