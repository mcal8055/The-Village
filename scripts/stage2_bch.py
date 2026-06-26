import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from stepmix.stepmix import StepMix
from paths import PROCESSED

WAVES={2:"Y1",3:"Y3",4:"Y5",5:"Y9"}
def build(parent):
    long=pd.read_csv(PROCESSED/f"holdout_{parent}_long.csv")
    wide=long.pivot_table(index="idnum",columns="wave",values="md_case_lib")
    wide.columns=[WAVES[c] for c in wide.columns]
    cov=long.sort_values("wave").drop_duplicates("idnum").set_index("idnum")[["poverty_cat","par_edu"]]
    df=wide.join(cov)
    df=df.dropna(subset=["poverty_cat","par_edu"])            # need covariates
    df=df[~df[list(WAVES.values())].isna().all(axis=1)]       # need >=1 depression wave
    X=df[list(WAVES.values())].astype(float)                  # binary indicators (NaN ok)
    Y=df[["poverty_cat","par_edu"]].astype(float)
    Y=(Y-Y.mean())/Y.std()                                    # z-score -> OR per SD
    return df,X,Y

for parent in ["mother","father"]:
    df,X,Y=build(parent)
    m=StepMix(n_components=2, measurement="binary_nan", structural="covariate",
              n_steps=3, correction="BCH", n_init=20, max_iter=2000, random_state=7)
    m.fit(X.values, Y.values)
    lab=m.predict(X.values, Y.values)
    par=m.get_parameters()
    # measurement profile: P(depressed) per wave per class -> identify persistent class
    meas=par["measurement"]["pis"] if "pis" in par.get("measurement",{}) else None
    # robust profile from observed data by assigned class
    prof=pd.DataFrame(X.values, columns=list(WAVES.values())).assign(c=lab).groupby("c").mean()
    persistent=int(prof.mean(axis=1).idxmax())
    props=pd.Series(lab).value_counts(normalize=True).sort_index()
    print(f"\n===== {parent.upper()} (holdout, n={len(df)}) =====")
    print("class proportions:", {int(k):round(v,3) for k,v in props.items()},
          "| persistent class =", persistent)
    print("measurement profile P(depressed) by class/wave:\n", prof.round(3).to_string())
    # structural betas: covariate -> class membership (log-odds), BCH-corrected
    st=par["structural"]
    print("structural param keys:", list(st.keys()))
    beta=st.get("beta")
    print("beta shape/val:\n", np.round(np.array(beta),3) if beta is not None else st)
