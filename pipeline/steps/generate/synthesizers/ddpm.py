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
an in-house diffusion baseline, not as the reference TabDDPM -- and it
stays that: this is a small, dependency-free reimplementation, not a
tuned competitor to the published models.

STABILIZERS (added after the first evaluation run, where the reverse
chain visibly diverged: numeric outputs piled up at the quantile
extremes and roughly half of them decoded to null, i.e. below the
sentinel floor):

  * x0-CLAMPING. Each reverse step now reconstructs the implied clean
    row x0_hat = (x_t - sqrt(1-ab_t) * eps_pred) / sqrt(ab_t), clamps it
    per dimension to the training embedding's own min/max, and rebuilds
    the posterior mean from the clamped value using the standard DDPM
    posterior. The naive epsilon-parameterized update this replaced had
    no mechanism to stop an over-confident noise prediction from walking
    the state outside the data manifold, and the error compounds over
    1000 steps. Clamping is the standard fix (it is what reference DDPM
    implementations do by default) and it is honest here: the bound is a
    property of the training embedding, which this non-DP model is
    allowed to depend on.
  * EMA WEIGHTS (decay 0.999). Sampling uses an exponential moving
    average of the denoiser weights rather than the last SGD iterate,
    which is standard practice for diffusion models and removes the
    step-to-step noise in the final parameters.
  * A LONGER BUDGET. See config.synthesizer_params['ddpm']: 500 epochs
    (37 s) left the model plainly under-trained; the default is now 4000
    (~5 min).

Everything else -- the cosine schedule, the one-hot relaxation, the
optional logic guidance -- is unchanged.
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
        # Per-dimension support of the training embedding, used to clamp
        # the implied x0 at every reverse step (see the module docstring).
        # Widened by a small margin so the clamp bounds the chain without
        # snapping every sample onto an observed extreme.
        span = x0.max(axis=0) - x0.min(axis=0)
        margin = 0.05 * np.maximum(span, 1e-3)
        self._x0_min = (x0.min(axis=0) - margin).astype(np.float32)
        self._x0_max = (x0.max(axis=0) + margin).astype(np.float32)

        # LOGIC-GUIDED SAMPLING (optional): mine boolean implications
        # from the training frame with the SAME code the coherence audit
        # uses, and store the one-hot index pairs needed to penalize
        # rule violations during sampling. The audit's instrument
        # becomes a generation-time prior; the fair evaluation bar
        # remains the holdout's own violation rate. Implication rules
        # only (the bulk of the rule set); numeric-typed rules are not
        # differentiable through one-hot blocks and are left to the
        # audit. Disclosed wherever results are reported: the guided
        # model is optimized toward rules mined from its training data.
        self._guidance = None
        g_scale = float(self.params.get("guidance_scale", 0) or 0)
        if g_scale > 0:
            from pipeline.steps.coherence.rules import mine_boolean_implications

            offsets = {}
            off = len(self._num_cols)
            for c in self._cat_cols:
                offsets[c] = (off, len(self._categories[c]))
                off += len(self._categories[c])

            def _cat_idx(col, wanted):
                for i, v in enumerate(self._categories[col]):
                    if v.strip().lower() == wanted:
                        return i
                return None

            pairs = []
            for r in mine_boolean_implications(df):
                a, b = r["if_true"], r["then_true"]
                if a in offsets and b in offsets:
                    ai = _cat_idx(a, "true")
                    bi = _cat_idx(b, "false")
                    if ai is not None and bi is not None:
                        pairs.append((offsets[a][0], offsets[a][1], ai,
                                      offsets[b][0], offsets[b][1], bi))
            if pairs:
                self._guidance = {"scale": g_scale, "pairs": pairs}
                print(f"  logic guidance ON: {len(pairs)} implication rules as a "
                      f"sampling-time prior (scale {g_scale}).")
            else:
                print("  logic guidance requested but no applicable implication "
                      "rules were mined -- sampling unguided.")

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

        # EMA of the weights: sampling from the last SGD iterate is
        # needlessly noisy, and averaging is the standard remedy for
        # diffusion models. Only floating-point tensors are averaged;
        # anything else (buffers of integer counters, if the module ever
        # grows one) is copied as-is.
        ema_decay = float(self.params.get("ema_decay", 0.999))
        ema = {k: v.detach().clone() for k, v in self._model.state_dict().items()}

        def _ema_update():
            with torch.no_grad():
                for k, v in self._model.state_dict().items():
                    if ema[k].is_floating_point():
                        ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                    else:
                        ema[k].copy_(v.detach())

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
                _ema_update()
            if (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"  ddpm epoch {epoch + 1}/{epochs} loss {loss.item():.4f}")

        # Sampling uses the EMA weights; the raw final iterate is not
        # kept, so what is saved to disk is exactly what generated the
        # published rows.
        self._model.load_state_dict(ema)
        self._model.eval()
        self._ema_decay = ema_decay
        print(f"  ddpm: sampling weights = EMA(decay {ema_decay}) of training weights.")

    def sample(self, n_rows: int) -> pd.DataFrame:
        import torch

        device = self._device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.params.get("seed", 0) or 0) + 1)
        betas = torch.from_numpy(self._betas).to(device)
        alphas = torch.from_numpy(self._alphas).to(device)
        ab = torch.from_numpy(self._alpha_bar).to(device)

        def _rule_loss(xg):
            # expected joint probability of (antecedent true AND
            # consequent explicitly false), summed over rules -- a
            # differentiable relaxation of the audit's violation count.
            import torch as _t

            loss = xg.new_zeros(())
            soft = {}
            for a_start, a_len, ai, b_start, b_len, bi in self._guidance["pairs"]:
                if (a_start, a_len) not in soft:
                    soft[(a_start, a_len)] = _t.softmax(
                        xg[:, a_start:a_start + a_len], dim=1)
                if (b_start, b_len) not in soft:
                    soft[(b_start, b_len)] = _t.softmax(
                        xg[:, b_start:b_start + b_len], dim=1)
                pa = soft[(a_start, a_len)][:, ai]
                pb = soft[(b_start, b_len)][:, bi]
                loss = loss + (pa * pb).sum()
            return loss

        x0_min = torch.from_numpy(self._x0_min).to(device)
        x0_max = torch.from_numpy(self._x0_max).to(device)

        x = torch.randn(n_rows, self._dim, generator=gen, device=device)
        with torch.no_grad():
            for t in range(self._T - 1, -1, -1):
                tt = torch.full((n_rows,), t, device=device, dtype=torch.float32)
                eps = self._model(x, tt / self._T)

                # x0-clamped posterior mean (see the module docstring).
                # ab_prev is 1 at t=0, which makes the final step reduce
                # to "return the clamped x0_hat" -- the standard
                # convention.
                ab_t = ab[t]
                ab_prev = ab[t - 1] if t > 0 else torch.ones_like(ab_t)
                x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                x0_hat = torch.clamp(x0_hat, min=x0_min, max=x0_max)
                x = ((ab_prev.sqrt() * betas[t] / (1 - ab_t)) * x0_hat
                     + (alphas[t].sqrt() * (1 - ab_prev) / (1 - ab_t)) * x)

                if self._guidance is not None:
                    with torch.enable_grad():
                        xg = x.detach().requires_grad_(True)
                        grad = torch.autograd.grad(_rule_loss(xg), xg)[0]
                    x = x - self._guidance["scale"] * grad
                if t > 0:
                    # sigma_t = sqrt(beta_t), unchanged: Ho et al. report
                    # this and the posterior variance as equivalent in
                    # practice, and keeping it isolates the effect of the
                    # x0 clamp on the same noise levels as before.
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
                                 "+ one-hot categoricals, MLP denoiser, x0-clamped "
                                 "posterior sampling, EMA weights)",
                  "timesteps": getattr(self, "_T", None),
                  "x0_clamped": hasattr(self, "_x0_min"),
                  "ema_decay": getattr(self, "_ema_decay", None),
                  "logic_guided": bool(getattr(self, "_guidance", None)),
                  "guidance_rules": len(self._guidance["pairs"]) if getattr(
                      self, "_guidance", None) else 0})
        return d
