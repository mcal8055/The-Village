import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from stepmix.stepmix import StepMix
from paths import PROCESSED, ANALYSIS_STAGE1
WAVES={2:"Y1",3:"Y3",4:"Y5",5:"Y9"}; IND=list(WAVES.values())
def build(parent):
    long=pd.read_csv(PROCESSED/f"holdout_{parent}_long.csv")
    wide=long.pivot_table(index="idnum",columns="wave",values="md_case_lib"); wide.columns=[WAVES[c] for c in wide.columns]
    cov=long.sort_values("wave").drop_duplicates("idnum").set_index("idnum")[["poverty_cat","par_edu"]]
    df=wide.join(cov).dropna(subset=["poverty_cat","par_edu"])
    df=df[~df[IND].isna().all(axis=1)]
    X=df[IND].astype(float).values; Y=df[["poverty_cat","par_edu"]].astype(float)
    Y=((Y-Y.mean())/Y.std()).values
    return X,Y
def fit_persistent_beta(X,Y,n_init=6,seed=0):
    m=StepMix(n_components=2,measurement="binary_nan",structural="covariate",n_steps=3,
              correction="BCH",n_init=n_init,max_iter=1500,random_state=seed,progress_bar=0,verbose=0)
    m.fit(X,Y); lab=m.predict(X,Y)
    prof=pd.DataFrame(X,columns=IND).assign(c=lab).groupby("c").mean().mean(axis=1)
    persistent=int(prof.idxmax())
    beta=np.array(m.get_parameters()["structural"]["beta"])
    return beta[persistent,1:], (lab==persistent).mean()   # [pov, edu] log-odds, persistent prop
res={}
for parent in ["mother","father"]:
    X,Y=build(parent)
    pt,prop=fit_persistent_beta(X,Y,n_init=20,seed=7)
    boots=[]
    rng=np.random.default_rng(1)
    for b in range(300):
        idx=rng.integers(0,len(X),len(X))
        try: bb,_=fit_persistent_beta(X[idx],Y[idx],n_init=4,seed=b); boots.append(bb)
        except Exception: pass
    boots=np.array(boots)
    out={"n":int(len(X)),"persistent_prop":round(float(prop),3),"n_boot":int(len(boots))}
    for j,name in enumerate(["poverty_cat","par_edu"]):
        lo,hi=np.percentile(boots[:,j],[2.5,97.5])
        out[name]={"OR_per_SD":round(float(np.exp(pt[j])),3),
                   "CI":[round(float(np.exp(lo)),3),round(float(np.exp(hi)),3)],
                   "logOR":round(float(pt[j]),3)}
    res[parent]=out
    print(parent.upper(),json.dumps(out))
json.dump(res,open(ANALYSIS_STAGE1/"stage2_bch_results.json","w"),indent=2)
print("\n[wrote analysis/stage1/stage2_bch_results.json]")
