import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from stepmix.stepmix import StepMix
from paths import PROCESSED, ANALYSIS_STAGE1
WAVES={2:"Y1",3:"Y3",4:"Y5",5:"Y9"}; IND=list(WAVES.values())
vuln=pd.read_csv(PROCESSED/"preexisting_vuln.csv")
def build(parent):
    long=pd.read_csv(PROCESSED/f"holdout_{parent}_long.csv")
    wide=long.pivot_table(index="idnum",columns="wave",values="md_case_lib"); wide.columns=[WAVES[c] for c in wide.columns]
    cov=long.sort_values("wave").drop_duplicates("idnum").set_index("idnum")[["poverty_cat","par_edu"]]
    preds=["poverty_cat","par_edu",f"fh_{parent}"]+(["cesd_father_baseline"] if parent=="father" else [])
    df=wide.join(cov).join(vuln.set_index("idnum")[[c for c in preds if c not in ("poverty_cat","par_edu")]]).dropna(subset=preds)
    df=df[~df[IND].isna().all(axis=1)]
    Y=df[preds].astype(float); Yz=((Y-Y.mean())/Y.std()).values
    return df[IND].astype(float).values, Yz, preds
def fit_beta(X,Y,n_init=4,seed=0):
    m=StepMix(n_components=2,measurement="binary_nan",structural="covariate",n_steps=3,
              correction="BCH",n_init=n_init,max_iter=1500,random_state=seed,progress_bar=0,verbose=0)
    m.fit(X,Y); lab=m.predict(X,Y)
    prof=pd.DataFrame(X,columns=IND).assign(c=lab).groupby("c").mean().mean(axis=1)
    p=int(prof.idxmax()); return np.array(m.get_parameters()["structural"]["beta"])[p,1:]
res={}
for parent in ["mother","father"]:
    X,Y,preds=build(parent); pt=fit_beta(X,Y,20,7)
    rng=np.random.default_rng(1); B=[]
    for b in range(250):
        idx=rng.integers(0,len(X),len(X))
        try:B.append(fit_beta(X[idx],Y[idx],4,b))
        except Exception:pass
    B=np.array(B); out={"n":len(X),"n_boot":len(B)}
    for j,nm in enumerate(preds):
        lo,hi=np.percentile(B[:,j],[2.5,97.5])
        out[nm]={"OR":round(float(np.exp(pt[j])),3),"CI":[round(float(np.exp(lo)),3),round(float(np.exp(hi)),3)]}
    res[parent]=out; print(parent.upper(),json.dumps(out))
json.dump(res,open(ANALYSIS_STAGE1/"stage2b_preexisting_results.json","w"),indent=2)
print("[done]")
