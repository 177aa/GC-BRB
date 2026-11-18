# -*- coding: utf-8 -*-
"""
GC-BRB Experiment D: strong conventional ML baselines on the exact 37-D evidence
================================================================================

Goal
----
Answer the reviewer question:
"Could ordinary black-box classifiers using exactly the same 37-D evidence
already match or exceed the full GC-BRB?"

Formal protocol
---------------
- SAME official 37-D evidence representation
- SAME predefined Train / Validation / Test split for every target/run
- TRAIN: fit
- VALIDATION: select hyperparameters / restart
- TEST: evaluation only
- no GC-BRB retraining
- no test-driven tuning

Baselines
---------
1. Logistic Regression
2. Random Forest
3. XGBoost
4. RBF-SVM
5. Shallow MLP 37 -> 16 -> 1

Official reference outputs on the same TEST genes
--------------------------------------------------
6. Residual Linear
7. GC-BRB Backbone
8. Final GC-BRB

Primary metric: AUPRC
Secondary metric: AUROC

Install:
    pip install numpy pandas scipy scikit-learn xgboost

Notes
-----
- Class imbalance is handled using TRAIN-only class weights / balanced loss.
- All scaling is fitted on TRAIN only.
- XGBoost scale_pos_weight is computed from TRAIN only.
- The six PPI targets are six views of one benchmark, not six fully
  independent datasets. Therefore output focuses on effect sizes,
  per-target consistency, and run-level win/loss rather than treating
  n=6 views as independent external-study replicates.
"""

from pathlib import Path
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "Experiment D requires xgboost. Install it with:\n"
        "pip install xgboost"
    ) from e

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(r"D:\PyCharm 2024.1\data_h5\GC_BRB_3LAYER_FINAL")
INPUT_ROOT = PROJECT_ROOT / "results" / "experiment_D_37d_inputs" / "official_exact"
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_D_strong_ml_baselines" / "official_exact"

TARGETS = ["CPDB","MULTINET","PCNET","STRINGdb","IREF","IREF_2015"]
RUNS = range(1,6)
FEATURES = [f"F{i:02d}" for i in range(1,38)]

BASELINE_METHODS = [
    "LOGISTIC_37D",
    "RANDOM_FOREST_37D",
    "XGBOOST_37D",
    "RBF_SVM_37D",
    "SHALLOW_MLP_37x16x1",
]
OFFICIAL_METHODS = [
    "RESIDUAL_LINEAR_OFFICIAL",
    "BACKBONE_OFFICIAL",
    "FINAL_GC_BRB_OFFICIAL",
]
ALL_METHODS = BASELINE_METHODS + OFFICIAL_METHODS

SEED_BASE = 20260810
RESUME = True

# Fixed, modest grids chosen before seeing TEST.
LOGISTIC_C_GRID = [0.01,0.1,1.0,10.0,100.0]

RF_CANDIDATES = [
    {"n_estimators":400,"max_depth":None,"min_samples_leaf":1,"max_features":"sqrt"},
    {"n_estimators":500,"max_depth":12,  "min_samples_leaf":1,"max_features":"sqrt"},
    {"n_estimators":500,"max_depth":12,  "min_samples_leaf":5,"max_features":"sqrt"},
    {"n_estimators":500,"max_depth":8,   "min_samples_leaf":5,"max_features":0.5},
    {"n_estimators":600,"max_depth":None,"min_samples_leaf":5,"max_features":0.5},
    {"n_estimators":600,"max_depth":8,   "min_samples_leaf":20,"max_features":0.5},
]

XGB_CANDIDATES = [
    {"n_estimators":400,"max_depth":3,"learning_rate":0.05,"min_child_weight":1},
    {"n_estimators":600,"max_depth":3,"learning_rate":0.03,"min_child_weight":1},
    {"n_estimators":400,"max_depth":5,"learning_rate":0.05,"min_child_weight":1},
    {"n_estimators":600,"max_depth":5,"learning_rate":0.03,"min_child_weight":1},
    {"n_estimators":500,"max_depth":3,"learning_rate":0.05,"min_child_weight":5},
    {"n_estimators":600,"max_depth":5,"learning_rate":0.03,"min_child_weight":5},
]

SVM_C_GRID = [0.5,2.0,8.0]
SVM_GAMMA_GRID = ["scale",0.03]

MLP_HIDDEN = 16
MLP_L2_GRID = [0.0,1e-4,1e-3,1e-2]
MLP_RESTARTS = 3
MLP_MAXITER = 400

TIE_TOL = 1e-12


# ============================================================
# Metrics / selection
# ============================================================

def metric_dict(y,score):
    y=np.asarray(y,int)
    score=np.asarray(score,float)
    if len(np.unique(y))<2:
        return {"AUPRC":np.nan,"AUROC":np.nan}
    return {
        "AUPRC":average_precision_score(y,score),
        "AUROC":roc_auc_score(y,score),
    }

def better(c,b):
    """Validation AUPRC first, exact tie -> AUROC."""
    if b is None:
        return True
    if c["AUPRC"] > b["AUPRC"] + TIE_TOL:
        return True
    if abs(c["AUPRC"]-b["AUPRC"]) <= TIE_TOL:
        return c["AUROC"] > b["AUROC"]
    return False


# ============================================================
# Input / audit
# ============================================================

def load_run(target,run):
    rn=f"run_{run:02d}"
    d=INPUT_ROOT/target/rn

    tr=pd.read_csv(d/"train_37d.csv")
    va=pd.read_csv(d/"val_37d.csv")
    te=pd.read_csv(d/"test_37d.csv")
    off=pd.read_csv(d/"official_test_predictions.csv")

    for df,name in [(tr,"train"),(va,"val"),(te,"test")]:
        missing=[c for c in ["GlobalIndex","Label"]+FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{target} {rn} {name} missing columns: {missing}")
        if not np.isfinite(df[FEATURES].to_numpy(float)).all():
            raise ValueError(f"{target} {rn} {name} contains non-finite features")

    si=[set(x.GlobalIndex.astype(int).tolist()) for x in [tr,va,te]]
    if si[0]&si[1] or si[0]&si[2] or si[1]&si[2]:
        raise RuntimeError(f"Split overlap: {target} {rn}")

    if not np.array_equal(
        te.GlobalIndex.to_numpy(int),
        off.GlobalIndex.to_numpy(int)
    ):
        raise RuntimeError(f"Official prediction order mismatch: {target} {rn}")

    if not np.array_equal(
        te.Label.to_numpy(int),
        off.Label.to_numpy(int)
    ):
        raise RuntimeError(f"Official label mismatch: {target} {rn}")

    return tr,va,te,off


# ============================================================
# Logistic Regression
# ============================================================

def fit_logistic(Xtr,ytr,Xv,yv,seed):
    scaler=StandardScaler()
    Xtrz=scaler.fit_transform(Xtr)
    Xvz=scaler.transform(Xv)

    best_model=None
    best_val=None
    best_C=None

    for C in LOGISTIC_C_GRID:
        model=LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
        model.fit(Xtrz,ytr)
        pv=model.predict_proba(Xvz)[:,1]
        mv=metric_dict(yv,pv)
        if better(mv,best_val):
            best_val=mv
            best_model=model
            best_C=C

    return {"model":best_model,"scaler":scaler,"C":best_C,"val":best_val}


# ============================================================
# Random Forest
# ============================================================

def fit_rf(Xtr,ytr,Xv,yv,seed):
    best_model=None
    best_val=None
    best_cfg=None

    for i,cfg in enumerate(RF_CANDIDATES):
        model=RandomForestClassifier(
            **cfg,
            class_weight="balanced_subsample",
            random_state=seed+100*i,
            n_jobs=-1,
        )
        model.fit(Xtr,ytr)
        pv=model.predict_proba(Xv)[:,1]
        mv=metric_dict(yv,pv)
        if better(mv,best_val):
            best_val=mv
            best_model=model
            best_cfg=dict(cfg)

    return {"model":best_model,"config":best_cfg,"val":best_val}


# ============================================================
# XGBoost
# ============================================================

def fit_xgb(Xtr,ytr,Xv,yv,seed):
    npos=max(int(np.sum(ytr==1)),1)
    nneg=max(int(np.sum(ytr==0)),1)
    spw=nneg/npos

    best_model=None
    best_val=None
    best_cfg=None

    for i,cfg in enumerate(XGB_CANDIDATES):
        model=XGBClassifier(
            **cfg,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=seed+1000*i,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(Xtr,ytr)
        pv=model.predict_proba(Xv)[:,1]
        mv=metric_dict(yv,pv)
        if better(mv,best_val):
            best_val=mv
            best_model=model
            best_cfg=dict(cfg)
            best_cfg["scale_pos_weight"]=spw

    return {"model":best_model,"config":best_cfg,"val":best_val}


# ============================================================
# RBF-SVM
# ============================================================

def fit_svm(Xtr,ytr,Xv,yv,seed):
    scaler=StandardScaler()
    Xtrz=scaler.fit_transform(Xtr)
    Xvz=scaler.transform(Xv)

    best_model=None
    best_val=None
    best_cfg=None

    for C in SVM_C_GRID:
        for gamma in SVM_GAMMA_GRID:
            model=SVC(
                C=C,
                gamma=gamma,
                kernel="rbf",
                class_weight="balanced",
                probability=False,
                random_state=seed,
                cache_size=2000,
            )
            model.fit(Xtrz,ytr)
            sv=model.decision_function(Xvz)
            mv=metric_dict(yv,sv)
            if better(mv,best_val):
                best_val=mv
                best_model=model
                best_cfg={"C":C,"gamma":gamma}

    return {"model":best_model,"scaler":scaler,"config":best_cfg,"val":best_val}


# ============================================================
# Custom shallow MLP 37 -> 16 -> 1, balanced BCE
# ============================================================

def sigmoid(z):
    z=np.asarray(z,float)
    out=np.empty_like(z)
    pos=z>=0
    out[pos]=1/(1+np.exp(-z[pos]))
    ez=np.exp(z[~pos])
    out[~pos]=ez/(1+ez)
    return out

def mlp_unpack(th,d,h):
    c=0
    W1=th[c:c+d*h].reshape(d,h); c+=d*h
    b1=th[c:c+h]; c+=h
    W2=th[c:c+h]; c+=h
    b2=th[c]
    return W1,b1,W2,b2

def mlp_predict(th,X,h=MLP_HIDDEN):
    W1,b1,W2,b2=mlp_unpack(th,X.shape[1],h)
    H=np.tanh(X@W1+b1)
    return sigmoid(H@W2+b2)

def fit_mlp_one(X,y,l2,seed):
    rng=np.random.default_rng(seed)
    n,d=X.shape
    h=MLP_HIDDEN

    W1=rng.normal(0,np.sqrt(2/(d+h)),size=(d,h))
    b1=np.zeros(h)
    W2=rng.normal(0,np.sqrt(2/(h+1)),size=h)
    prior=np.clip(y.mean(),1e-6,1-1e-6)
    b2=np.array([math.log(prior/(1-prior))])
    th0=np.r_[W1.ravel(),b1,W2,b2]

    pos=y==1
    neg=y==0
    wp=0.5/max(int(pos.sum()),1)
    wn=0.5/max(int(neg.sum()),1)
    sw=np.where(pos,wp,wn)

    def fg(th):
        W1,b1,W2,b2=mlp_unpack(th,d,h)
        H=np.tanh(X@W1+b1)
        p=sigmoid(H@W2+b2)
        pc=np.clip(p,1e-7,1-1e-7)

        loss=-np.sum(sw*(y*np.log(pc)+(1-y)*np.log(1-pc)))
        loss+=0.5*l2*(np.sum(W1*W1)+np.sum(W2*W2))

        dz=sw*(p-y)
        gW2=H.T@dz+l2*W2
        gb2=dz.sum()
        dA=(dz[:,None]*W2[None,:])*(1-H*H)
        gW1=X.T@dA+l2*W1
        gb1=dA.sum(0)

        grad=np.r_[gW1.ravel(),gb1,gW2,gb2]
        return loss,grad

    r=minimize(
        lambda z:fg(z)[0],
        th0,
        jac=lambda z:fg(z)[1],
        method="L-BFGS-B",
        options={"maxiter":MLP_MAXITER,"ftol":1e-10,"maxls":40},
    )
    if not np.isfinite(r.fun):
        raise RuntimeError("MLP optimization failed")
    return r.x

def fit_mlp(Xtr,ytr,Xv,yv,seed):
    scaler=StandardScaler()
    Xtrz=scaler.fit_transform(Xtr)
    Xvz=scaler.transform(Xv)

    best=None
    best_val=None

    for li,l2 in enumerate(MLP_L2_GRID):
        for restart in range(MLP_RESTARTS):
            s=seed+10000*li+restart
            th=fit_mlp_one(Xtrz,ytr,l2,s)
            pv=mlp_predict(th,Xvz)
            mv=metric_dict(yv,pv)
            if better(mv,best_val):
                best_val=mv
                best={
                    "theta":th,
                    "l2":l2,
                    "restart":restart+1,
                    "scaler":scaler,
                    "val":mv,
                }

    return best


# ============================================================
# Run
# ============================================================

def run_one(target,run):
    rn=f"run_{run:02d}"
    out_dir=OUT_ROOT/target/rn
    out_dir.mkdir(parents=True,exist_ok=True)

    result_path=out_dir/"D_run_results.csv"
    pred_path=out_dir/"D_test_predictions.csv"
    hp_path=out_dir/"D_selected_hyperparameters.json"

    if RESUME and result_path.exists() and pred_path.exists() and hp_path.exists():
        print(f"[{target} {rn}] resume")
        return pd.read_csv(result_path).to_dict("records")

    tr,va,te,off=load_run(target,run)

    Xtr=tr[FEATURES].to_numpy(float)
    ytr=tr.Label.to_numpy(int)
    Xv=va[FEATURES].to_numpy(float)
    yv=va.Label.to_numpy(int)
    Xte=te[FEATURES].to_numpy(float)
    yte=te.Label.to_numpy(int)

    seed=SEED_BASE + 1000*TARGETS.index(target) + run

    selected={}
    predictions={}

    print(f"[{target} {rn}] Logistic")
    m=fit_logistic(Xtr,ytr,Xv,yv,seed)
    predictions["LOGISTIC_37D"]=m["model"].predict_proba(
        m["scaler"].transform(Xte))[:,1]
    selected["LOGISTIC_37D"]={"C":m["C"],"Validation":m["val"]}

    print(f"[{target} {rn}] RandomForest")
    m=fit_rf(Xtr,ytr,Xv,yv,seed)
    predictions["RANDOM_FOREST_37D"]=m["model"].predict_proba(Xte)[:,1]
    selected["RANDOM_FOREST_37D"]={**m["config"],"Validation":m["val"]}

    print(f"[{target} {rn}] XGBoost")
    m=fit_xgb(Xtr,ytr,Xv,yv,seed)
    predictions["XGBOOST_37D"]=m["model"].predict_proba(Xte)[:,1]
    selected["XGBOOST_37D"]={**m["config"],"Validation":m["val"]}

    print(f"[{target} {rn}] RBF-SVM")
    m=fit_svm(Xtr,ytr,Xv,yv,seed)
    predictions["RBF_SVM_37D"]=m["model"].decision_function(
        m["scaler"].transform(Xte))
    selected["RBF_SVM_37D"]={**m["config"],"Validation":m["val"]}

    print(f"[{target} {rn}] Shallow MLP")
    m=fit_mlp(Xtr,ytr,Xv,yv,seed)
    predictions["SHALLOW_MLP_37x16x1"]=mlp_predict(
        m["theta"],m["scaler"].transform(Xte))
    selected["SHALLOW_MLP_37x16x1"]={
        "hidden":MLP_HIDDEN,
        "l2":m["l2"],
        "restart":m["restart"],
        "Validation":m["val"],
    }

    predictions["RESIDUAL_LINEAR_OFFICIAL"]=off.ResidualLinearProbability.to_numpy(float)
    predictions["BACKBONE_OFFICIAL"]=off.BackboneProbability.to_numpy(float)
    predictions["FINAL_GC_BRB_OFFICIAL"]=off.FinalProbability.to_numpy(float)

    rows=[]
    for method in ALL_METHODS:
        mm=metric_dict(yte,predictions[method])
        rows.append({
            "Target":target,
            "Run":run,
            "Method":method,
            "TestN":len(yte),
            "Positive":int(np.sum(yte==1)),
            "Negative":int(np.sum(yte==0)),
            "AUPRC":mm["AUPRC"],
            "AUROC":mm["AUROC"],
        })

    rdf=pd.DataFrame(rows)
    rdf.to_csv(result_path,index=False)

    pdf=pd.DataFrame({
        "GlobalIndex":te.GlobalIndex.to_numpy(int),
        "Label":yte,
    })
    for method in ALL_METHODS:
        pdf[method]=predictions[method]
    pdf.to_csv(pred_path,index=False)

    hp_path.write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")

    return rows


# ============================================================
# Summary
# ============================================================

def summarize(run_df):
    target=(
        run_df.groupby(["Target","Method"],as_index=False)
        .agg(
            AUPRC_Mean=("AUPRC","mean"),
            AUPRC_Std=("AUPRC","std"),
            AUROC_Mean=("AUROC","mean"),
            AUROC_Std=("AUROC","std"),
        )
    )

    macro=(
        target.groupby("Method",as_index=False)
        .agg(
            MacroAUPRC=("AUPRC_Mean","mean"),
            AcrossTargetStdAUPRC=("AUPRC_Mean","std"),
            MacroAUROC=("AUROC_Mean","mean"),
        )
        .sort_values("MacroAUPRC",ascending=False)
    )

    # Pair each method against FINAL GC-BRB.
    final=run_df[run_df.Method=="FINAL_GC_BRB_OFFICIAL"].sort_values(["Target","Run"])
    comp=[]

    for method in [m for m in ALL_METHODS if m!="FINAL_GC_BRB_OFFICIAL"]:
        b=run_df[run_df.Method==method].sort_values(["Target","Run"])
        if not np.array_equal(
            final[["Target","Run"]].to_numpy(),
            b[["Target","Run"]].to_numpy()
        ):
            raise RuntimeError("Pairing mismatch")

        da=final.AUPRC.to_numpy()-b.AUPRC.to_numpy()
        du=final.AUROC.to_numpy()-b.AUROC.to_numpy()

        target_da=[]
        target_du=[]
        for target_name in TARGETS:
            mask=final.Target.eq(target_name).to_numpy()
            target_da.append(float(np.mean(da[mask])))
            target_du.append(float(np.mean(du[mask])))

        comp.append({
            "Baseline":method,
            "MacroDeltaAUPRC_FinalMinusBaseline":float(np.mean(target_da)),
            "MinTargetDeltaAUPRC":float(np.min(target_da)),
            "MaxTargetDeltaAUPRC":float(np.max(target_da)),
            "TargetsFinalHigherAUPRC":int(np.sum(np.asarray(target_da)>0)),
            "TargetsBaselineHigherAUPRC":int(np.sum(np.asarray(target_da)<0)),
            "RunWinsFinal_AUPRC":int(np.sum(da>1e-10)),
            "RunTies_AUPRC":int(np.sum(np.abs(da)<=1e-10)),
            "RunLossesFinal_AUPRC":int(np.sum(da<-1e-10)),
            "MacroDeltaAUROC_FinalMinusBaseline":float(np.mean(target_du)),
        })

    return target,macro,pd.DataFrame(comp)


def main():
    OUT_ROOT.mkdir(parents=True,exist_ok=True)

    audit_path=INPUT_ROOT/"experiment_D_export_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(
            "Run export_gcbrb_37d_baseline_inputs.m first:\n"+str(audit_path)
        )

    audit=pd.read_csv(audit_path)
    if not (
        audit["AllFinite"].astype(bool).all()
        and audit["SplitsDisjoint"].astype(bool).all()
        and audit["OfficialTestAligned"].astype(bool).all()
    ):
        raise RuntimeError("Experiment D export audit failed.")

    all_rows=[]

    print("="*78)
    print("GC-BRB Experiment D: same-37D strong ML baselines")
    print("TRAIN fit / VALIDATION select / TEST frozen")
    print("="*78)

    for target in TARGETS:
        for run in RUNS:
            rows=run_one(target,run)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(
                OUT_ROOT/"D_all_run_results_partial.csv",index=False)

    run_df=pd.DataFrame(all_rows)
    run_df.to_csv(OUT_ROOT/"D_all_run_results.csv",index=False)

    target,macro,comp=summarize(run_df)
    target.to_csv(OUT_ROOT/"D_target_summary.csv",index=False)
    macro.to_csv(OUT_ROOT/"D_macro_summary.csv",index=False)
    comp.to_csv(OUT_ROOT/"D_vs_final_comparisons.csv",index=False)

    lines=[
        "GC-BRB Experiment D: same-37D strong ML baseline audit",
        "="*76,
        "",
        "Protocol:",
        "- Exact 37-D evidence exported from each official evidence_v3layer.mat.",
        "- Exact predefined Train / Validation / Test split.",
        "- TRAIN used for fitting.",
        "- VALIDATION used for hyperparameter/restart selection.",
        "- TEST excluded from fitting and selection.",
        "- Class imbalance handled from TRAIN only.",
        "- Official residual/backbone/final predictions aligned by GlobalIndex.",
        "- Primary metric: AUPRC; secondary: AUROC.",
        "- Six PPI targets are six benchmark views, not six independent datasets;",
        "  comparisons emphasize effect sizes and consistency rather than n=6 inferential p-values.",
        "",
        "Macro summary:",
        macro.to_string(index=False),
        "",
        "Final GC-BRB comparisons:",
        comp.to_string(index=False),
    ]
    (OUT_ROOT/"EXPERIMENT_D_AUDIT.txt").write_text(
        "\n".join(lines),encoding="utf-8")

    print("\nMacro summary:")
    print(macro.to_string(index=False))
    print("\nSaved to:")
    print(OUT_ROOT)
    print("\nPlease upload:")
    for fn in [
        "EXPERIMENT_D_AUDIT.txt",
        "D_all_run_results.csv",
        "D_target_summary.csv",
        "D_macro_summary.csv",
        "D_vs_final_comparisons.csv",
    ]:
        print(" -",fn)


if __name__=="__main__":
    main()
