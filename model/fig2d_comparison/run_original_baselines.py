#!/usr/bin/env python3
"""Original model families/grids on verified GG_OOF_1_0 portable evidence.

Fit Train only; select using Validation group AP, then AUROC (original rule).
Test is used only after selection. MLP uses the intended balanced logistic
loss in stable logaddexp form with its matching gradient, rather than clipping
the loss while leaving the gradient unclipped. Original architecture, L2 grid,
restart seeds, optimizer, and 400-iteration budget are retained and recorded.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,importlib.util,time,warnings,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.io import loadmat
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'same_input_baselines_20261024/runtime_deps'))
import xgboost
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'baseline_results'
TARGETS=['CPDB','MULTINET','PCNET','STRINGdb','IREF','IREF_2015']
SOURCE=ROOT/'original_sources/original_python/GC_BRB_ExperimentD_37d_strong_ml.py'
spec=importlib.util.spec_from_file_location('original_models',SOURCE)
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
METHODS=['LOGISTIC_37D','RANDOM_FOREST_37D','XGBOOST_37D','RBF_SVM_37D','SHALLOW_MLP_37x16x1']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def writej(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def grouped(y,score,g):
    unique,inv=np.unique(g,return_inverse=True);counts=np.bincount(inv)
    yy=np.bincount(inv,weights=y)/counts
    assert np.isin(yy,[0,1]).all()
    return yy.astype(int),np.bincount(inv,weights=score)/counts,unique
def metric(y,score,g):
    yy,ss,_=grouped(y,score,g)
    return dict(AUPRC=float(average_precision_score(yy,ss)),AUROC=float(roc_auc_score(yy,ss)))
def better(v,b):
    return b is None or v['AUPRC']>b['AUPRC']+1e-12 or (abs(v['AUPRC']-b['AUPRC'])<=1e-12 and v['AUROC']>b['AUROC'])

def mlp_fg(theta,X,y,l2):
    W1,b1,W2,b2=orig.mlp_unpack(theta,X.shape[1],16)
    H=np.tanh(X@W1+b1);z=H@W2+b2;p=expit(z)
    sw=np.where(y==1,.5/np.sum(y==1),.5/np.sum(y==0))
    loss=np.sum(sw*(np.logaddexp(0,z)-y*z))+.5*l2*(np.sum(W1*W1)+np.sum(W2*W2))
    dz=sw*(p-y);dA=dz[:,None]*W2[None,:]*(1-H*H)
    grad=np.r_[(X.T@dA+l2*W1).ravel(),dA.sum(axis=0),H.T@dz+l2*W2,dz.sum()]
    return float(loss),grad

def fit_mlp(X,y,l2,seed):
    rng=np.random.default_rng(seed);d=X.shape[1];h=16
    W1=rng.normal(0,np.sqrt(2/(d+h)),size=(d,h));W2=rng.normal(0,np.sqrt(2/(h+1)),size=h)
    prior=np.clip(y.mean(),1e-6,1-1e-6)
    initial=np.r_[W1.ravel(),np.zeros(h),W2,np.log(prior/(1-prior))]
    result=minimize(mlp_fg,initial,args=(X,y,l2),jac=True,method='L-BFGS-B',
        options=dict(maxiter=400,ftol=1e-10,maxls=40))
    assert np.isfinite(result.fun) and np.isfinite(result.x).all()
    return result

def one(target,run):
    folder=OUT/target/f'run_{run:02d}';folder.mkdir(parents=True,exist_ok=True)
    final=folder/'status.json';fingerprint=sha(ROOT/'plan.json')
    if final.exists():
        status=json.loads(final.read_text());assert status['PlanSHA256']==fingerprint
        if status['Completed']:return status
    t=time.time();path=ROOT/'models'/target/f'run_{run:02d}/final_portable.mat'
    s=loadmat(path,simplify_cells=True)['state']
    assert s['base_fingerprint']=='cd7518374c6dc760e164a6c7e03ec0f7c89f939b8073f7704d7f4eb7a65a1704'
    ids=pd.read_csv(ROOT/'models'/target/'identity_record_map.csv')
    F=s['evidence']['feature_matrix'];y=ids.Label.to_numpy();groups=ids.GroupID.to_numpy(int)
    tr=s['fit_idx'].astype(int)-1;va=s['stop_idx'].astype(int)-1;te=s['held_idx'].astype(int)-1
    assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    assert F.shape==(22659,37) and np.isfinite(F).all()
    X,ytr=F[tr],y[tr].astype(int);Xv,yv=F[va],y[va].astype(int)
    scaler=StandardScaler().fit(X);XZ=scaler.transform(X);XVZ=scaler.transform(Xv)
    seed=20260810+1000*TARGETS.index(target)+run
    all_scores=[];search=[];selected=[];results=[]
    for method in METHODS:
        best=None;winner=None;best_model=None
        if method=='LOGISTIC_37D':configs=[dict(C=c) for c in orig.LOGISTIC_C_GRID]
        elif method=='RANDOM_FOREST_37D':configs=orig.RF_CANDIDATES
        elif method=='XGBOOST_37D':configs=orig.XGB_CANDIDATES
        elif method=='RBF_SVM_37D':configs=[dict(C=c,gamma=g) for c,g in itertools.product(orig.SVM_C_GRID,orig.SVM_GAMMA_GRID)]
        else:configs=[dict(l2=l,restart=r+1) for l in orig.MLP_L2_GRID for r in range(3)]
        for i,cfg in enumerate(configs):
            opt={};warning_messages=[]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if method=='LOGISTIC_37D':
                    model=LogisticRegression(C=cfg['C'],solver='lbfgs',class_weight='balanced',max_iter=3000,random_state=seed)
                    model.fit(XZ,ytr);pv=model.predict_proba(XVZ)[:,1]
                    opt=dict(Iterations=int(model.n_iter_.max()))
                elif method=='RANDOM_FOREST_37D':
                    model=RandomForestClassifier(**cfg,class_weight='balanced_subsample',random_state=seed+100*i,n_jobs=1)
                    model.fit(X,ytr);pv=model.predict_proba(Xv)[:,1]
                elif method=='XGBOOST_37D':
                    model=XGBClassifier(**cfg,objective='binary:logistic',eval_metric='logloss',subsample=.8,colsample_bytree=.8,
                        reg_lambda=1.,scale_pos_weight=float((ytr==0).sum()/(ytr==1).sum()),random_state=seed+1000*i,n_jobs=1,tree_method='hist',verbosity=0)
                    model.fit(X,ytr);pv=model.predict_proba(Xv)[:,1]
                elif method=='RBF_SVM_37D':
                    model=SVC(**cfg,kernel='rbf',class_weight='balanced',probability=False,random_state=seed,cache_size=256)
                    model.fit(XZ,ytr);pv=model.decision_function(XVZ)
                    opt=dict(FitStatus=int(model.fit_status_))
                else:
                    li=orig.MLP_L2_GRID.index(cfg['l2']);ss=seed+10000*li+cfg['restart']-1
                    fit=fit_mlp(XZ,ytr,cfg['l2'],ss);model=fit.x
                    pv=orig.mlp_predict(model,XVZ)
                    opt=dict(Iterations=int(fit.nit),Success=bool(fit.success),Status=int(fit.status),Message=str(fit.message),
                        TrainLoss=float(fit.fun),GradientInfinityNorm=float(np.abs(fit.jac).max()))
                warning_messages=[str(w.message) for w in caught]
            v=metric(yv,pv,groups[va]);row=dict(Target=target,Run=run,Method=method,Trial=i+1,
                Params=json.dumps(cfg,sort_keys=True),ValidationAUPRC=v['AUPRC'],ValidationAUROC=v['AUROC'],
                Warnings='; '.join(warning_messages),**opt)
            search.append(row)
            if better(v,best):best=v;winner=row;best_model=model
        # No Test labels are used until the winner has been fixed.
        if method in ['LOGISTIC_37D','RBF_SVM_37D','SHALLOW_MLP_37x16x1']:
            full=scaler.transform(F)
        else:full=F
        if method=='RBF_SVM_37D':score=best_model.decision_function(full)
        elif method=='SHALLOW_MLP_37x16x1':score=orig.mlp_predict(best_model,full)
        else:score=best_model.predict_proba(full)[:,1]
        assert np.isfinite(score).all();all_scores.append(score)
        selected.append(winner)
        test=metric(y[te].astype(int),score[te],groups[te])
        results.append(dict(Target=target,Run=run,Method=method,**test))
        print(target,run,method,'completed',round(time.time()-t,1),'s',flush=True)
    pd.DataFrame(search).to_csv(folder/'validation_search.csv',index=False)
    pd.DataFrame(selected).to_csv(folder/'selected_parameters.csv',index=False)
    pd.DataFrame(results).to_csv(folder/'test_metrics.csv',index=False)
    scores=np.column_stack(all_scores)
    np.savez_compressed(folder/'all_record_scores.npz',GlobalIndex=np.arange(1,len(F)+1),GroupID=groups,
        Label=y,Methods=np.array(METHODS),Score=scores,TrainIndex=tr+1,ValidationIndex=va+1,TestIndex=te+1)
    pred=pd.DataFrame(dict(GlobalIndex=te+1,GroupID=groups[te],Label=y[te].astype(int)))
    for j,m in enumerate(METHODS):pred[m]=scores[te,j]
    pred.to_csv(folder/'test_predictions.csv',index=False)
    status=dict(Completed=True,Target=target,Run=run,PlanSHA256=fingerprint,InputMAT_SHA256=sha(path),
        TestUsedForSelection=False,ValidationSelection='group_AP_then_AUROC_then_first_grid_order',
        MLPStableObjectiveCorrection=True,Seconds=time.time()-t)
    writej(final,status);return status

def main():
    OUT.mkdir(exist_ok=True)
    plan_path=ROOT/'plan.json'
    if not plan_path.exists():
        plan=dict(Protocol='GG_OOF_1_0',Source=SOURCE.name,SourceSHA256=sha(SOURCE),RunnerSHA256=sha(Path(__file__)),
            Methods=METHODS,RFGrid=orig.RF_CANDIDATES,XGBGrid=orig.XGB_CANDIDATES,LogisticC=orig.LOGISTIC_C_GRID,
            SVMC=orig.SVM_C_GRID,SVMGamma=orig.SVM_GAMMA_GRID,MLP=dict(Hidden=16,Activation='tanh',Optimizer='L-BFGS-B',
                L2=orig.MLP_L2_GRID,Restarts=3,MaxIter=400,ftol=1e-10,maxls=40,
                Loss='sum balanced class weights * (logaddexp(0,z)-y*z) + .5*l2*weight_squares',
                Correction='Original source clipped p in loss but not gradient; stable intended BCE is used with matching analytic gradient'),
            Selection='Validation identity-group AP; within1e-12 tie AUROC; first grid order if tied',
            TestBasedSelection=False,SeedBase=20260810,Workers=6,TrainOnlyScaling=True,
            PriorAdamAndXGBDiagnosticResultsRetained=True,
            Versions=dict(Python=sys.version,NumPy=np.__version__,SciPy=scipy.__version__,ScikitLearn=sklearn.__version__,XGBoost=xgboost.__version__))
        # Gradient QA covers saturated logits as well as ordinary initialization.
        rng=np.random.default_rng(20261025);xx=rng.normal(size=(23,37));yy=np.arange(23)%2
        tt=rng.normal(scale=.4,size=37*16+16+16+1);tt[-1]=20
        f,g=mlp_fg(tt,xx,yy,.001);inds=rng.choice(len(tt),30,replace=False)
        errs=[]
        for k in inds:
            u=tt.copy();v=tt.copy();u[k]+=1e-5;v[k]-=1e-5
            num=(mlp_fg(u,xx,yy,.001)[0]-mlp_fg(v,xx,yy,.001)[0])/2e-5
            errs.append(abs(num-g[k]))
        assert max(errs)<1e-7
        plan['MLPGradientCheckMaxAbsError']=max(errs)
        writej(plan_path,plan)
    else:assert json.loads(plan_path.read_text())['RunnerSHA256']==sha(Path(__file__))
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(one,t,r) for t in TARGETS for r in range(1,6)]
        for job in as_completed(jobs):
            status=job.result();print('RUN COMPLETE',status['Target'],status['Run'],flush=True)
    frames=[pd.read_csv(f) for f in OUT.glob('*/run_*/test_metrics.csv')]
    all_rows=pd.concat(frames,ignore_index=True);assert len(all_rows)==150
    all_rows.to_csv(OUT/'all_model_runs.csv',index=False)
    bytask=all_rows.groupby(['Target','Method'],as_index=False).agg(AUPRC_Mean=('AUPRC','mean'),AUPRC_Std=('AUPRC','std'),AUROC_Mean=('AUROC','mean'),AUROC_Std=('AUROC','std'))
    bytask.to_csv(OUT/'task_summary.csv',index=False)
    bytask.groupby('Method',as_index=False).agg(MacroAUPRC=('AUPRC_Mean','mean'),MacroAUROC=('AUROC_Mean','mean')).to_csv(OUT/'macro_summary.csv',index=False)
    print('ALL ORIGINAL-FAMILY BASELINES COMPLETE',flush=True)

if __name__=='__main__':main()
