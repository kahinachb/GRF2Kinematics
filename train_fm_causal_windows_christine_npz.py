"""Flow Matching causal: GRFM[t-W+1:t] -> q[t] sur les NPZ corriges.

Chaque taille de ``--windows`` entraine un modele independant. Le Transformer
encode uniquement l'historique disponible; aucune GRFM future n'est fournie.
"""
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_linear_christine_npz import (
    JOINT_NAMES, load_pair, split_files, stats, correlation_columns,
    discover_variant_files,
)

MZ_INDICES = (5, 11)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class CausalWindowDataset(Dataset):
    def __init__(self, paths, window, xm, xs, ym, ys, stride=10,
                 first_frame=None, ignore_mz=False):
        self.window, self.ignore_mz = window, ignore_mz
        self.xm, self.xs = xm.astype(np.float32), xs.astype(np.float32)
        self.ym, self.ys = ym.astype(np.float32), ys.astype(np.float32)
        self.data, self.indices = [], []
        start = max(window - 1, first_frame or 0)
        for path in paths:
            x, y = load_pair(path, "q")
            item = (x.astype(np.float32), y.astype(np.float32))
            file_index = len(self.data); self.data.append(item)
            self.indices.extend((file_index, t) for t in range(start, len(x), stride))

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        file_index, t = self.indices[index]
        x, y = self.data[file_index]
        condition = ((x[t-self.window+1:t+1] - self.xm) / self.xs).copy()
        if self.ignore_mz: condition[:, MZ_INDICES] = 0
        target = (y[t] - self.ym) / self.ys
        return torch.from_numpy(condition), torch.from_numpy(target)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) *
                         (-math.log(10000) / max(half-1, 1)))
        emb = t[:, None] * 1000 * freq[None]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class CausalFlowModel(nn.Module):
    def __init__(self, max_window, dim=128, heads=4, layers=3):
        super().__init__()
        self.cond_in = nn.Linear(12, dim)
        self.pos = nn.Parameter(torch.randn(1, max_window, dim) * .01)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=.1,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.x_in = nn.Linear(29, dim)
        self.time = nn.Sequential(TimeEmbedding(dim), nn.Linear(dim, dim), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(dim*3, dim*2), nn.SiLU(),
                                 nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, 29))

    def forward(self, x_t, t, condition):
        # Tous les tokens de condition sont <= temps courant: pas de fuite future.
        memory = self.encoder(self.cond_in(condition) + self.pos[:, :condition.shape[1]])
        context = memory[:, -1]
        return self.out(torch.cat((self.x_in(x_t), self.time(t), context), dim=1))


class EMA:
    def __init__(self, model, decay=.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if torch.is_floating_point(v): self.shadow[k].lerp_(v.detach(), 1-self.decay)
            else: self.shadow[k].copy_(v)


@torch.no_grad()
def heun(model, condition, steps):
    x = torch.randn((len(condition), 29), device=condition.device)
    dt = 1 / steps
    for i in range(steps):
        t0 = torch.full((len(x),), i/steps, device=x.device)
        t1 = torch.full((len(x),), (i+1)/steps, device=x.device)
        v0 = model(x, t0, condition)
        xp = x + dt*v0
        x += .5*dt*(v0 + model(xp, t1, condition))
    return x


def evaluate(model, loader, device, ym, ys, steps):
    model.eval(); refs=[]; preds=[]
    with torch.no_grad():
        for condition, target in loader:
            condition = condition.to(device)
            prediction = heun(model, condition, steps).cpu().numpy()
            refs.append(target.numpy()); preds.append(prediction)
    ref = np.concatenate(refs)*ys + ym
    pred = np.concatenate(preds)*ys + ym
    rmse = np.degrees(np.sqrt(np.mean((pred-ref)**2, axis=0)))
    return rmse, correlation_columns(ref, pred)


def summary(rmse, cc):
    return np.mean(rmse), np.sqrt(np.mean(rmse**2)), np.nanmedian(cc), np.nanmean(cc)


def train_one(args, window, train, val, test_file, stats_values, device):
    xm,xs,ym,ys,_ = stats_values
    common = dict(window=window, xm=xm, xs=xs, ym=ym, ys=ys,
                  stride=args.frame_stride, first_frame=args.first_eval_frame,
                  ignore_mz=args.ignore_mz)
    train_ds = CausalWindowDataset(train, **common)
    val_ds = CausalWindowDataset(val, **common)
    synth_ds = CausalWindowDataset([test_file], **common)
    # Les champs reference du meme NPZ sont charges explicitement puis injectes.
    rx,ry = load_pair(test_file, "q", reference=True)
    ref_ds = CausalWindowDataset([], **common)
    ref_ds.data=[(rx.astype(np.float32),ry.astype(np.float32))]
    start=max(window-1,args.first_eval_frame)
    ref_ds.indices=[(0,t) for t in range(start,len(rx),args.frame_stride)]
    loader_kw=dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=torch.cuda.is_available())
    train_loader=DataLoader(train_ds,shuffle=True,drop_last=True,**loader_kw)
    val_loader=DataLoader(val_ds,shuffle=False,**loader_kw)
    synth_loader=DataLoader(synth_ds,shuffle=False,**loader_kw)
    ref_loader=DataLoader(ref_ds,shuffle=False,**loader_kw)
    model=CausalFlowModel(window,args.dim,args.heads,args.layers).to(device)
    ema=EMA(model,args.ema); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,
                                                   weight_decay=args.weight_decay)
    best=float("inf"); out=args.output_dir/f"window_{window}"; out.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(args.epochs):
        model.train(); total=0
        for condition,target in train_loader:
            condition,target=condition.to(device),target.to(device)
            x0=torch.randn_like(target); t=torch.rand(len(target),device=device)
            xt=t[:,None]*target+(1-t[:,None])*x0
            loss=nn.functional.mse_loss(model(xt,t,condition),target-x0)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step(); ema.update(model); total+=loss.item()
        raw={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema.shadow); model.eval(); vtotal=0
        with torch.no_grad():
            for bi,(condition,target) in enumerate(val_loader):
                condition,target=condition.to(device),target.to(device)
                gen=torch.Generator(device=device).manual_seed(args.seed+bi)
                x0=torch.randn(target.shape,generator=gen,device=device)
                t=torch.rand(len(target),generator=gen,device=device)
                xt=t[:,None]*target+(1-t[:,None])*x0
                vtotal+=nn.functional.mse_loss(model(xt,t,condition),target-x0).item()
        vloss=vtotal/max(len(val_loader),1); tloss=total/max(len(train_loader),1)
        if vloss<best:
            best=vloss; torch.save({"model":model.state_dict(),"window":window,"epoch":epoch,
                                    "stats":{k:v.tolist() for k,v in zip(("xm","xs","ym","ys"),(xm,xs,ym,ys))}},
                                   out/"best.pth")
        model.load_state_dict(raw); history.append((epoch,tloss,vloss))
        print(f"W={window:3d} epoch={epoch:03d} train={tloss:.5f} val={vloss:.5f}")
    checkpoint=torch.load(out/"best.pth",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"])
    srmse,scc=evaluate(model,synth_loader,device,ym,ys,args.steps)
    rrmse,rcc=evaluate(model,ref_loader,device,ym,ys,args.steps)
    np.savetxt(out/"history.csv",history,delimiter=",",header="epoch,train_loss,val_loss",comments="")
    with (out/"metrics_per_dof.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["dof","synth_rmse_deg","synth_cc","real_rmse_deg","real_cc"])
        w.writerows(zip(JOINT_NAMES,srmse,scc,rrmse,rcc))
    return summary(srmse,scc)+summary(rrmse,rcc)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=Path("DATA/Christine_synthetic"))
    p.add_argument("--output-dir",type=Path,default=Path("results_fm_causal_windows"))
    p.add_argument("--windows",type=int,nargs="+",default=[1,25,50,100,200])
    p.add_argument("--first-eval-frame",type=int,default=199)
    p.add_argument("--frame-stride",type=int,default=10)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-2)
    p.add_argument("--ema",type=float,default=.999); p.add_argument("--steps",type=int,default=20)
    p.add_argument("--dim",type=int,default=128); p.add_argument("--heads",type=int,default=4)
    p.add_argument("--layers",type=int,default=3); p.add_argument("--workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--ignore-mz",action=argparse.BooleanOptionalAction,default=False)
    return p.parse_args()


def main():
    args=parse_args(); seed_all(args.seed); args.output_dir.mkdir(parents=True,exist_ok=True)
    files=discover_variant_files(args.data_root); train,val,test=split_files(files,args.seed,.7,.15)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values=stats(train,"q"); rows=[]
    print(f"device={device} workers={args.workers} train={len(train)} val={len(val)} "
          f"test={len(test)} test_file={test[0].name}")
    for window in args.windows:
        result=train_one(args,window,train,val,test[0],values,device)
        rows.append((window,window*.01,*result))
    header="window_frames,window_seconds,synth_rmse_mean,synth_rmse_global,synth_cc_median,synth_cc_mean,real_rmse_mean,real_rmse_global,real_cc_median,real_cc_mean"
    np.savetxt(args.output_dir/"summary.csv",rows,delimiter=",",header=header,comments="")

if __name__=="__main__": main()
