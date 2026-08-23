"""
Native tabular denoising-diffusion synthesizer (TabDDPM-style, non-DP).

Self-contained on purpose: the established diffusion packages drag in
dependency trees we will not risk against a frozen campaign environment,
so this is a minimal, honest implementation of the standard recipe --
Gaussian DDPM over a continuous embedding of the table:

  * numeric columns  -> rank/quantile-normalized to N(0,1) (exactly the
    transform validated for the quantile-variant runs), diffused
    directly, inverted onto the empirical quantiles at decode (which
    also keeps every sampled value inside the observed support);
  * categorical columns (their "Missing" category included) -> one-hot
    blocks, diffused as continuous relaxations, decoded by per-block
    argmax so outputs are always real categories in the real spelling.

An MLP predicts the added noise given the noisy row and a Fourier time
embedding; sampling is standard ancestral DDPM. Reported in the paper as
an in-house diffusion baseline, not as the reference TabDDPM.
"""

import numpy as np
import pandas as pd

from pipeline.steps.generate.synthesizers.base import Synthesizer


class DDPMSynthesizer(Synthesizer):
    name = "ddpm"
    is_dp = False
    uses_gpu = True

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        import torch
        from sklearn.preprocessing import QuantileTransformer

        seed = int(self.params.get("seed", 0) or 0)
        torch.manual_seed(seed)
        self._num_cols = list(continuous_columns)
        self._cat_cols = list(categorical_columns)

        parts = []
        if self._num_cols:
            num = df[self._num_cols].to_numpy(dtype=float)
            if np.isnan(num).any():
                raise ValueError("ddpm expects the sentinel-encoded frame (no numeric nulls)")
            self._qt = QuantileTransformer(output_distribution="normal",
                                           n_quantiles=min(1000, len(df)),
                                           subsample=10 ** 9, random_state=seed)
            parts.append(self._qt.fit_transform(num))
        else:
            self._qt = None

        self._categories = {}
        for c in self._cat_cols:
            col = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
            cats = sorted(col.unique())
            self._categories[c] = cats
            onehot = np.zeros((len(df), len(cats)))
            index = {v: i for i, v in enumerate(cats)}
            onehot[np.arange(len(df)), col.map(index).to_numpy()] = 1.0
            parts.append(onehot)

        x0 = np.concatenate(parts, axis=1).astype(np.float32)
        self._dim = x0.shape[1]

        # cosine noise schedule
        self._T = int(self.params.get("timesteps", 1000))
        s = 0.008
        t = np.arange(self._T + 1) / self._T
        f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        self._betas = betas.astype(np.float32)
        self._alphas = (1.0 - betas).astype(np.float32)
        self._alpha_bar = np.cumprod(self._alphas).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        hidden = int(self.params.get("hidden", 512))
        t_dim = 64

        class Denoiser(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.freqs = torch.nn.Parameter(
                    torch.randn(t_dim // 2) * 4.0, requires_grad=False)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(dim + t_dim, hidden), torch.nn.SiLU(),
                    torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                    torch.nn.Linear(hidden, dim),
                )

            def forward(self, x, t_frac):
                ang = t_frac[:, None] * self.freqs[None, :] * 2 * np.pi
                temb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
                return self.net(torch.cat([x, temb], dim=1))

        self._model = Denoiser(self._dim).to(device)
        opt = torch.optim.Adam(self._model.parameters(),
                               lr=float(self.params.get("lr", 1e-3)))
        x0_t = torch.from_numpy(x0).to(device)
        ab = torch.from_numpy(self._alpha_bar).to(device)
        n = len(x0_t)
        batch = min(int(self.params.get("batch_size", 500)), n)
        epochs = int(self.params.get("epochs", 500))

        self._model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                xb = x0_t[idx]
                tt = torch.randint(0, self._T, (len(xb),), device=device)
                a = ab[tt][:, None]
                noise = torch.randn_like(xb)
                xt = a.sqrt() * xb + (1 - a).sqrt() * noise
                pred = self._model(xt, tt.float() / self._T)
                loss = torch.nn.functional.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"  ddpm epoch {epoch + 1}/{epochs} loss {loss.item():.4f}")
        self._model.eval()

    def sample(self, n_rows: int) -> pd.DataFrame:
        import torch

        device = self._device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.params.get("seed", 0) or 0) + 1)
        betas = torch.from_numpy(self._betas).to(device)
        alphas = torch.from_numpy(self._alphas).to(device)
        ab = torch.from_numpy(self._alpha_bar).to(device)

        x = torch.randn(n_rows, self._dim, generator=gen, device=device)
        with torch.no_grad():
            for t in range(self._T - 1, -1, -1):
                tt = torch.full((n_rows,), t, device=device, dtype=torch.float32)
                eps = self._model(x, tt / self._T)
                coef = betas[t] / (1 - ab[t]).sqrt()
                x = (x - coef * eps) / alphas[t].sqrt()
                if t > 0:
                    x = x + betas[t].sqrt() * torch.randn(
                        n_rows, self._dim, generator=gen, device=device)
        x = x.cpu().numpy()

        out = {}
        offset = 0
        if self._num_cols:
            k = len(self._num_cols)
            inv = self._qt.inverse_transform(x[:, :k].astype(np.float64))
            for i, c in enumerate(self._num_cols):
                out[c] = inv[:, i]
            offset = k
        for c in self._cat_cols:
            cats = self._categories[c]
            block = x[:, offset:offset + len(cats)]
            out[c] = [cats[i] for i in block.argmax(axis=1)]
            offset += len(cats)
        return pd.DataFrame(out)[self._num_cols + self._cat_cols]

    def describe(self) -> dict:
        d = super().describe()
        d.update({"model_class": "native Gaussian DDPM (quantile-normalized numerics "
                                 "+ one-hot categoricals, MLP denoiser)",
                  "timesteps": getattr(self, "_T", None)})
        return d
