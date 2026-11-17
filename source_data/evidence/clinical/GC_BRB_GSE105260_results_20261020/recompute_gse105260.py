"""Recompute eight-CpG statistics from retained values or the original GEO matrix.

The selected-table mode reproduces statistics, not extraction from the full matrix.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,hashlib,json,platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu,norm

PROBES=['cg01942646','cg06650260','cg07914457','cg10055471','cg16177693','cg16762386','cg16985652','cg24580782']
SCRIPT_REGIONS={'cg01942646':'TSS200','cg06650260':"1stExon;5'UTR",'cg07914457':'TSS1500','cg10055471':'1stExon','cg16177693':'TSS200','cg16762386':"1stExon;5'UTR",'cg16985652':'TSS1500','cg24580782':'1stExon'}
def bh(p):
 p=np.asarray(p,float);o=np.argsort(p);n=len(p);q=np.empty(n);q[o]=np.minimum(1,np.minimum.accumulate((p[o]*n/np.arange(1,n+1))[::-1])[::-1]);return q

def read_matrix(matrix):
 meta={};hits={};nrows=0;in_table=False;header=None
 with gzip.open(matrix,'rt',encoding='utf-8-sig',errors='strict',newline='') as f:
  for line in f:
   if line.startswith('!series_matrix_table_begin'):in_table=True;continue
   if line.startswith('!series_matrix_table_end'):break
   if not in_table:
    if line.startswith('!'):
     z=next(csv.reader([line.rstrip('\r\n')],delimiter='\t'));meta.setdefault(z[0],[]).append(z[1:])
    continue
   z=next(csv.reader([line.rstrip('\r\n')],delimiter='\t'))
   if header is None:header=z;continue
   nrows+=1
   if z[0] in PROBES:
    assert z[0] not in hits,'Duplicate selected probe '+z[0];hits[z[0]]=np.array(z[1:],float)
 assert set(hits)==set(PROBES) and header[0]=='ID_REF'
 gsms=meta['!Sample_geo_accession'][0];titles=meta['!Sample_title'][0];assert len(set(gsms))==len(gsms)==44
 assert set(header[1:])==set(gsms)
 groups=['Normal' if s.startswith('Normal Renal') else 'Primary' if s.startswith('Primary ccRCC') else 'Metastasis' if s.startswith('Metastasis ccRCC') else 'Other' for s in titles]
 assert dict(Counter(groups))=={'Metastasis':26,'Normal':9,'Primary':9}
 # Verify the tissue-characteristic labels independently from titles.
 tissues=[row for row in meta['!Sample_characteristics_ch1'] if all(s.startswith('tissue:') for s in row)];assert len(tissues)==1
 expected={'Normal':'tissue: Normal renal tissue','Primary':'tissue: Primary RCC tumor','Metastasis':'tissue: Metastasis RCC tumor'}
 assert all(t==expected[g] for t,g in zip(tissues[0],groups))
 order=[header.index(s)-1 for s in gsms];X=np.column_stack([hits[p][order] for p in PROBES]);assert X.shape==(44,8) and np.isfinite(X).all() and ((X>=0)&(X<=1)).all()
 S=pd.DataFrame({'GSM':gsms,'Title':titles,'TissueCharacteristic':tissues[0],'Group':groups})
 for j,p in enumerate(PROBES):S[p]=X[:,j]
 S['EightCpGMeanBeta']=X.mean(axis=1)
 return S, meta, nrows

def read_selected(path):
 S=pd.read_csv(path)
 required=['GSM','Title','TissueCharacteristic','Group']+PROBES
 if not set(required).issubset(S.columns):raise ValueError('Selected table is missing required columns.')
 if len(S)!=44 or not S.GSM.is_unique:raise ValueError('Expected 44 distinct GEO samples.')
 expected={'Normal':'tissue: Normal renal tissue','Primary':'tissue: Primary RCC tumor','Metastasis':'tissue: Metastasis RCC tumor'}
 inferred=['Normal' if s.startswith('Normal Renal') else 'Primary' if s.startswith('Primary ccRCC') else 'Metastasis' if s.startswith('Metastasis ccRCC') else 'Other' for s in S.Title]
 if not np.array_equal(inferred,S.Group):raise ValueError('Sample titles disagree with group labels.')
 if S.Group.value_counts().to_dict()!={'Metastasis':26,'Normal':9,'Primary':9}:raise ValueError('Unexpected sample-group counts.')
 if not all(t==expected[g] for t,g in zip(S.TissueCharacteristic,S.Group)):raise ValueError('Tissue labels disagree with groups.')
 X=S[PROBES].to_numpy(float)
 if not np.isfinite(X).all() or not ((X>=0)&(X<=1)).all():raise ValueError('Invalid beta values.')
 means=X.mean(axis=1)
 if 'EightCpGMeanBeta' in S and not np.allclose(S.EightCpGMeanBeta,means,rtol=0,atol=1e-12):raise ValueError('Stored sample means disagree with the eight probes.')
 S['EightCpGMeanBeta']=means
 return S

def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def analyze(S,out,input_path,input_mode,meta,nrows):
 out.mkdir(exist_ok=True,parents=True)
 X=S[PROBES].to_numpy(float);groups=S.Group.tolist()
 S.to_csv(out/'GSE105260_selected_CpG_sample_values.csv',index=False)
 rows=[];independent=[]
 for p in PROBES:
  v={g:S.loc[S.Group==g,p].to_numpy(float) for g in ['Normal','Primary','Metastasis']};N=v['Normal'];T=v['Primary'];M=v['Metastasis'];m=len(T);n=len(N)
  result=mannwhitneyu(T,N,alternative='two-sided',method='asymptotic',use_continuity=True)
  auto=mannwhitneyu(T,N,alternative='two-sided',method='auto')
  # Independent U and asymptotic p from pair comparisons and tie-corrected variance.
  dif=T[:,None]-N[None,:];u=float(((dif>0)+.5*(dif==0)).sum());combined=np.r_[T,N];ct=np.unique(combined,return_counts=True)[1];V=m*n/12*((m+n+1)-np.sum(ct**3-ct)/((m+n)*(m+n-1)));z=(abs(u-m*n/2)-.5)/np.sqrt(V);pind=min(1,2*norm.sf(z))
  assert abs(u-result.statistic)<1e-12 and abs(pind-result.pvalue)<1e-12 and abs(auto.pvalue-result.pvalue)<1e-12
  independent.append({'CpG':p,'U':u,'asymptotic_p_independent':pind,'has_ties':bool((ct>1).any())})
  rows.append({'CpG':p,'Region_as_encoded_in_supplied_script':SCRIPT_REGIONS[p],'N_Normal':n,'N_Primary':m,'N_Metastasis':len(M),'MedianBeta_Normal':np.median(N),'MedianBeta_Primary':np.median(T),'MedianBeta_Metastasis':np.median(M),'DeltaBeta_Primary_minus_Normal':np.median(T)-np.median(N),'U_Primary_vs_Normal':result.statistic,'RankBiserial_Primary_vs_Normal':2*u/(m*n)-1,'P_Primary_vs_Normal':result.pvalue})
 C=pd.DataFrame(rows);C['BH_FDR_8CpGs']=bh(C.P_Primary_vs_Normal);C.to_csv(out/'GSE105260_CpG_statistics.csv',index=False)
 V={g:S.loc[S.Group==g,'EightCpGMeanBeta'].to_numpy(float) for g in ['Normal','Primary','Metastasis']}
 primary=mannwhitneyu(V['Primary'],V['Normal'],alternative='two-sided',method='asymptotic',use_continuity=True);metast=mannwhitneyu(V['Metastasis'],V['Normal'],alternative='two-sided',method='asymptotic',use_continuity=True)
 G={'N_Normal':len(V['Normal']),'N_Primary':len(V['Primary']),'N_Metastasis':len(V['Metastasis']),'MedianEightCpGMeanBeta_Normal':np.median(V['Normal']),'MedianEightCpGMeanBeta_Primary':np.median(V['Primary']),'MedianEightCpGMeanBeta_Metastasis':np.median(V['Metastasis']),'DeltaMedian_Primary_minus_Normal':np.median(V['Primary'])-np.median(V['Normal']),'Primary_vs_Normal_U':primary.statistic,'Primary_vs_Normal_rank_biserial':2*primary.statistic/81-1,'Primary_vs_Normal_P_nominal':primary.pvalue,'Metastasis_vs_Normal_U':metast.statistic,'Metastasis_vs_Normal_P_nominal':metast.pvalue}
 pd.DataFrame([G]).to_csv(out/'GSE105260_eight_CpG_mean_statistics.csv',index=False)

 summary={'input':input_path.name,'input_sha256':file_hash(input_path),'input_mode':input_mode,'matrix_probe_rows':nrows,'sample_count':44,'selected_numeric_values':int(X.size),'group_counts':dict(Counter(groups)),'metadata_tissue_matches_titles':True,'data_processing':list(dict.fromkeys(x for row in meta.get('!Sample_data_processing',[]) for x in row)),'primary_BH_significant_CpGs':C.loc[C.BH_FDR_8CpGs<.05,'CpG'].tolist(),'sample_mean_statistics':G,'independent_U_p_checks':independent,'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'annotation_scope':'Region labels retain the supplied script annotations. The array manifest is required to independently verify transcript, gene and region assignments.'}
 (out/'recomputation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 if meta:(out/'GEO_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 return C,summary

def main():
 root=Path(__file__).resolve().parent
 p=argparse.ArgumentParser(description=__doc__)
 sources=p.add_mutually_exclusive_group()
 sources.add_argument('--matrix',type=Path,help='Original GEO series matrix; validates probe extraction.')
 sources.add_argument('--selected-values',type=Path,help='Retained eight-CpG sample table.')
 p.add_argument('--output',type=Path,default=root/'recomputed')
 args=p.parse_args()
 if args.matrix:
  path=args.matrix.resolve();S,meta,nrows=read_matrix(path);mode='full_series_matrix'
 else:
  path=(args.selected_values or root/'GSE105260_selected_CpG_sample_values.csv').resolve()
  S=read_selected(path);meta={};nrows=None;mode='selected_probe_table'
 C,summary=analyze(S,args.output.resolve(),path,mode,meta,nrows)
 print(C.to_string(index=False))
 print('COMPLETED:',mode,'|',args.output.resolve())

if __name__=='__main__':main()
