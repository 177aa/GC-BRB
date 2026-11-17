"""Summarize saved predictions; no training, fitting or inference is performed."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import beta
BASE=Path(__file__).resolve().parent

def ci(k,n):
    return (0.0 if k==0 else beta.ppf(.025,k,n-k+1),1.0 if k==n else beta.ppf(.975,k+1,n-k))

def main():
    runs=pd.read_csv(BASE/'inputs/all_run_results.csv')
    candidates=pd.read_csv(BASE/'inputs/all13603_group_scores_and_new_membership.csv')
    val=[];test=[]
    for _,s in runs.iterrows():
        valid=bool(s.HPThresholdValid);common={'Task':s.Target,'Run':int(s.Run),'Seed':int(s.Seed),'ThresholdValid':valid,'Threshold':s.HPThreshold if valid else np.nan}
        n=int(s.HPValidationCalls) if valid else 0;k=round(n*s.HPValidationPrecision) if valid else 0;lo,hi=ci(k,n) if valid else (np.nan,np.nan)
        val.append({**common,'TP':k if valid else np.nan,'FP':n-k if valid else np.nan,'Calls':n if valid else np.nan,'Precision':k/n if valid else np.nan,'CI95_Lower':lo,'CI95_Upper':hi})
        d=pd.read_csv(BASE/f'inputs/{s.Target}/run_{int(s.Run):02d}/test_group_predictions.csv')
        assert d.GroupID.is_unique
        npos=int((d.Label==1).sum())
        if valid:
            assert abs(d.FinalProbability-s.HPThreshold).min()>1e-8
            pred=d.FinalProbability>=s.HPThreshold;tp=int((pred&(d.Label==1)).sum());fp=int((pred&(d.Label==0)).sum());calls=tp+fp;lo,hi=ci(tp,calls)
            metrics={'TP':tp,'FP':fp,'Calls':calls,'Precision':tp/calls,'CI95_Lower':lo,'CI95_Upper':hi,'Recall':tp/npos}
        else:metrics={k:np.nan for k in ['TP','FP','Calls','Precision','CI95_Lower','CI95_Upper','Recall']}
        test.append({**common,'N_Positive':npos,**metrics})
    denom=int(runs.HPThresholdValid.sum());mainset=set(candidates.loc[candidates.Tier1,'GroupID']);rows=[]
    for rate in [.70,.75,.80,.85,.90,.95,1.0]:
        k=math.ceil(rate*denom-1e-10);sub=candidates[candidates.SupportCount>=k];ids=set(sub.GroupID)
        rows.append({'SupportRate':rate,'ValidThresholdModels':denom,'MinCount':k,'Candidates':len(ids),'OverlapWithTier1':len(ids&mainset),'JaccardWithTier1':len(ids&mainset)/len(ids|mainset),'DBUnion':int(sub[['Hit_NCG','Hit_OncoKB','Hit_COSMIC_CGC']].any(axis=1).sum()),'IntOGenNew':int(sub.IntOGenNewMembership.sum())})
    pd.DataFrame(val).to_csv(BASE/'validation_thresholds_and_precision_intervals.csv',encoding='utf-8-sig',index=False)
    pd.DataFrame(test).to_csv(BASE/'test_results_at_fixed_thresholds.csv',encoding='utf-8-sig',index=False)
    pd.DataFrame(rows).to_csv(BASE/'support_rate_sensitivity.csv',encoding='utf-8-sig',index=False)
    print(f'Summarized {len(runs)} saved models; {denom} valid thresholds; main Tier-1 {len(mainset)}.')

if __name__=='__main__':main()
