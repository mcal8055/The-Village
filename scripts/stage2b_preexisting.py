import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from stepmix.stepmix import StepMix
from paths import PROCESSED
WAVES={2:"Y1",3:"Y3",4:"Y5",5:"Y9"}; IND=list(WAVES.values())
vuln=pd.read_csv(PROCESSED/"preexisting_vuln.csv")
def build(parent):
    long=pd.read_csv(PROCESSED/f"holdout_{parent}_long.csv")
    wide=long.pivot_table(index="idnum",columns="wave",values="md_case_lib"); wide.columns=[WAVES[c] for c in wide.columns]
    cov=long.sort_values("wave").drop_duplicates("idnum").set_index("idnum")[["poverty_cat","par_edu"]]
    preds=["poverty_cat","par_edu", f"fh_{parent}"] + (["cesd_father_baseline"] if parent=="father" else [])
    df=wide.join(cov).join(vuln.set_index("idnum")[[c for c in preds if c not in ("poverty_cat","par_edu")]])
    df=df.dropna(subset=preds)
    df=df[~df[IND].isna().all(axis=1)]
    X=df[IND].astype(float).values
    Y=df[preds].astype(float); Yz=((Y-Y.mean())/Y.std())
    return X, Yz.values, preds, len(df)
for parent in ["mother","father"]:
    X,Y,preds,n=build(parent)
    m=StepMix(n_components=2,measurement="binary_nan",structural="covariate",n_steps=3,
              correction="BCH",n_init=20,max_iter=2000,random_state=7,progress_bar=0,verbose=0)
    m.fit(X,Y); lab=m.predict(X,Y)
    prof=pd.DataFrame(X,columns=IND).assign(c=lab).groupby("c").mean().mean(axis=1)
    persistent=int(prof.idxmax())
    beta=np.array(m.get_parameters()["structural"]["beta"])[persistent,1:]
    print(f"\n===== {parent.upper()} (holdout, n={n}, persistent={ (lab==persistent).mean()*100:.0f}% ) =====")
    print(" predictor              OR per SD (persistent-class membership)")
    for name,b in zip(preds,beta):
        print(f"  {name:22} {np.exp(b):.3f}   (logOR {b:+.3f})")
