import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from paths import PROCESSED
P=pd.read_csv(PROCESSED/"village_panel.csv")
CORE=["v_instrumental","v_grandparent","v_moves"]; CONF=["fh","poverty_cat","par_edu"]
for parent in ["mother","father"]:
    d=P[P.parent==parent].copy()
    feats_extra=["v_kintrust","v_partnercare"] if parent=="mother" else []
    facets=CORE+feats_extra
    use=["dep_t","dep_prev","wave","idnum"]+facets+CONF
    d=d[use].dropna()
    for c in ["v_instrumental","v_moves","poverty_cat","par_edu"]+([ "v_kintrust","v_partnercare"] if parent=="mother" else []):
        d[c]=(d[c]-d[c].mean())/d[c].std()
    print(f"\n############## {parent.upper()}  (n_rows={len(d)}, n_fam={d.idnum.nunique()}) ##############")

    # ---- GEE (panel-inferential) ----
    inter=" + ".join(f"{f}:fh" for f in facets)
    f="dep_t ~ dep_prev + wave + "+" + ".join(facets+["fh","poverty_cat","par_edu"])+" + "+inter
    res=smf.gee(f,"idnum",data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    OR=np.exp(res.params); CI=np.exp(res.conf_int()); pv=res.pvalues
    print("  -- GEE logistic (OR [95% CI], cluster-robust) --")
    for term in facets+["fh"]+[f"{x}:fh" for x in facets]:
        if term in OR.index:
            star="*" if pv[term]<.05 else " "
            print(f"   {term:24} OR {OR[term]:.3f} [{CI.loc[term,0]:.3f},{CI.loc[term,1]:.3f}] p={pv[term]:.3f}{star}")

    # ---- EBM (shape functions + grouped CV) ----
    X=d[["dep_prev","wave"]+facets+["fh","poverty_cat","par_edu"]]; y=d["dep_t"].astype(int); g=d["idnum"]
    fi={c:i for i,c in enumerate(X.columns)}
    pairs=[(fi[x],fi["fh"]) for x in facets]
    aucs=[]
    for tr,te in GroupKFold(5).split(X,y,g):
        mm=ExplainableBoostingClassifier(interactions=pairs,random_state=0).fit(X.iloc[tr],y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te],mm.predict_proba(X.iloc[te])[:,1]))
    ebm=ExplainableBoostingClassifier(interactions=pairs,random_state=0).fit(X,y)
    print(f"  -- EBM: person-grouped 5-fold AUC = {np.mean(aucs):.3f} (sd {np.std(aucs):.3f}) --")
    imp=pd.Series(dict(zip(ebm.term_names_,ebm.term_importances()))).sort_values(ascending=False)
    print("   term importance (top 8):")
    for t,v in imp.head(8).items(): print(f"     {t:30} {v:.4f}")
    # shape direction for village facets (score at low vs high bin)
    print("   village shape direction (Δlog-odds low→high):")
    for t in facets:
        idx=ebm.term_names_.index(t); sc=np.array(ebm.term_scores_[idx])
        core=sc[1:-1] if len(sc)>2 else sc
        print(f"     {t:18} {core[0]:+.3f} -> {core[-1]:+.3f}  (range {core.max()-core.min():.3f})")
