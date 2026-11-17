"""Recompute six-gene univariable Cox results from retained expression or GEO inputs.

The retained expression table is the default input. Use --from-raw to repeat
probe mapping, log2(MAS5 + 1) transformation and complete-case selection.
Statistical calculations are retained from the previously supplied script.
"""
import argparse, csv, gzip, hashlib, io, json, os, platform, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=root)
parser.add_argument('--output-dir',type=Path,default=root/'recomputed')
parser.add_argument('--from-raw',action='store_true')
args=parser.parse_args()
input_dir=args.input_dir.resolve();output_dir=args.output_dir.resolve()
output_dir.mkdir(parents=True,exist_ok=True)
f=input_dir/'GSE68465_series_matrix.txt.gz'
a=input_dir/'GPL96_annotation.txt'
out=output_dir/'GSE68465_FROZEN6_OS_COX.csv'
outx=output_dir/'GSE68465_FROZEN6_EXPRESSION.csv'
genes=['RPS6KA1','IL2RB','NR0B2','CBFA2T2','HSPG2','PRKDC']

def parse_tsv(s):
    return [x.strip().strip('"') for x in next(csv.reader([s], delimiter='\t'))]

if args.from_raw:
    assert os.path.exists(f), "STOP missing GSE68465 matrix"
    assert os.path.exists(a), "STOP missing GPL96 annotation"

    def parse_tsv(s):
        return [x.strip().strip('"') for x in next(csv.reader([s], delimiter="\t"))]

    # ---------- read GEO series matrix ----------
    with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
        lines = [x.rstrip("\r\n") for x in fh]

    geo = next((x for x in lines if x.startswith("!Sample_geo_accession")), None)
    assert geo is not None, "STOP no !Sample_geo_accession"
    samples = parse_tsv(geo)[1:]
    assert len(samples) == 462, f"STOP expected 462 GEO samples, got {len(samples)}"
    assert len(set(samples)) == 462, "STOP duplicate GSM IDs"

    meta = {}

    source = next((x for x in lines if x.startswith("!Sample_source_name_ch1")), None)
    if source is not None:
        meta["source"] = parse_tsv(source)[1:]

    for line in lines:
        if not line.startswith("!Sample_characteristics_ch1"):
            continue
        v = parse_tsv(line)[1:]
        if not v or ":" not in v[0]:
            continue
        key = v[0].split(":",1)[0].strip().lower()
        vals = [x.split(":",1)[1].strip() if ":" in x else "" for x in v]
        meta[key] = vals

    for key in ["disease_state","vital_status","months_to_last_contact_or_death"]:
        assert key in meta, f"STOP missing metadata field: {key}"
        assert len(meta[key]) == 462, f"STOP {key} length={len(meta[key])}, expected 462"

    b = next(i for i,x in enumerate(lines) if x.startswith("!series_matrix_table_begin"))
    e = next(i for i,x in enumerate(lines) if x.startswith("!series_matrix_table_end"))
    assert e > b + 2, "STOP malformed series matrix table"

    E = pd.read_csv(
        io.StringIO("\n".join(lines[b+1:e])),
        sep="\t",
        index_col=0
    )
    E.index = E.index.astype(str).str.strip().str.strip('"')
    E.columns = E.columns.astype(str).str.strip().str.strip('"')

    assert E.shape[1] == 462, f"STOP expression columns={E.shape[1]}, expected 462"
    assert set(E.columns) == set(samples), "STOP GSM mismatch between expression and metadata"
    E = E.loc[:, samples]

    print("SERIES_MATRIX_ROWS=", E.shape[0])
    print("SERIES_MATRIX_SAMPLES=", E.shape[1])

    # ---------- read GPL96 annotation ----------
    with open(a, "rt", encoding="utf-8", errors="replace") as fh:
        alines = [x.rstrip("\r\n") for x in fh]

    ab = next((i for i,x in enumerate(alines) if "platform_table_begin" in x.lower()), None)
    ae = next((i for i,x in enumerate(alines) if "platform_table_end" in x.lower()), None)

    assert ab is not None and ae is not None, "STOP GPL96 platform table markers not found"
    assert ae > ab + 2, "STOP GPL96 platform table empty"

    P = pd.read_csv(
        io.StringIO("\n".join(alines[ab+1:ae])),
        sep="\t",
        dtype=str
    ).fillna("")

    idc = next((c for c in P.columns if c.strip().upper() == "ID"), None)
    symc = next(
        (c for c in P.columns if c.strip().lower().replace("_"," ") == "gene symbol"),
        None
    )

    assert idc is not None, "STOP GPL96 ID column not found"
    assert symc is not None, "STOP GPL96 Gene Symbol column not found; columns=" + repr(P.columns.tolist())

    P[idc] = P[idc].astype(str).str.strip()
    P[symc] = P[symc].astype(str)

    mapping = {}

    for g in genes:
        def contains_gene(s):
            parts = [z.strip().upper() for z in re.split(r"\s*///\s*", str(s)) if z.strip()]
            return g in parts

        probes = P.loc[P[symc].map(contains_gene), idc].tolist()
        probes = [p for p in probes if p in E.index]
        probes = list(dict.fromkeys(probes))
        mapping[g] = probes

    missing = [g for g in genes if len(mapping[g]) == 0]
    assert not missing, "STOP frozen genes without GPL96 probes: " + repr(missing)

    print("PROBE_MAP:")
    for g in genes:
        print(g, mapping[g])

    # ---------- collapse all probes BEFORE survival analysis ----------
    # fixed rule: log2(MAS5 + 1), then mean across ALL mapped probes
    M = pd.DataFrame(index=samples)

    for g in genes:
        q = E.loc[mapping[g]].apply(pd.to_numeric, errors="coerce")
        assert np.isfinite(q.to_numpy(dtype=float)).all(), "STOP invalid expression values: " + g
        logq = np.log2(q + 1.0)
        M[g] = logq.mean(axis=0).reindex(samples)

    # ---------- clinical table ----------
    D = pd.DataFrame({
        "GSM": samples,
        "Disease": meta["disease_state"],
        "OS_status": meta["vital_status"],
        "OS_months_raw": meta["months_to_last_contact_or_death"],
        "Site": meta.get("source", [""] * 462)
    }).set_index("GSM")

    D = D.join(M)

    D = D.loc[
        D["Disease"].astype(str).str.strip().str.lower().eq("lung adenocarcinoma")
    ].copy()

    assert len(D) == 443, f"STOP expected 443 GEO LUAD arrays before OS filtering, got {len(D)}"

    D["event"] = (
        D["OS_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"dead":1, "alive":0})
    )

    D["time"] = pd.to_numeric(
        D["OS_months_raw"].replace({"--":np.nan, "":np.nan, "na":np.nan, "NA":np.nan}),
        errors="coerce"
    )

    D = D.loc[
        D["event"].notna() &
        D["time"].notna() &
        (D["time"] > 0)
    ].copy()

    assert len(D) == 442, f"STOP expected 442 complete-case OS LUAD after excluding missing OS, got {len(D)}"
    assert int(D["event"].sum()) > 100, f"STOP too few deaths: {int(D['event'].sum())}"

    print("LUAD_COMPLETE_CASE_EXPECTED=442")
    print("LUAD_USABLE_OS=", len(D))
    print("OS_EVENTS=", int(D["event"].sum()))


else:
    expr=input_dir/'GSE68465_FROZEN6_EXPRESSION.csv'
    probes=input_dir/'GSE68465_probe_mapping.csv'
    if not expr.is_file():raise FileNotFoundError('Required retained input: '+str(expr))
    if not probes.is_file():raise FileNotFoundError('Required probe mapping: '+str(probes))
    D=pd.read_csv(expr).set_index('GSM')
    required=['Site','OS_status','time','event']+genes
    if not set(required).issubset(D.columns):raise ValueError('Missing expression or survival columns.')
    if len(D)!=442 or not D.index.is_unique:raise ValueError('Expected 442 distinct complete-case samples.')
    if not np.isfinite(D[['time','event']+genes].to_numpy(float)).all():raise ValueError('Nonfinite input values.')
    if not (D.time>0).all() or not D.event.isin([0,1]).all():raise ValueError('Invalid survival outcomes.')
    if int(D.event.sum())!=236:raise ValueError('Expected 236 events in the retained cohort.')
    expected_event=D.OS_status.astype(str).str.strip().str.lower().map({'dead':1,'alive':0})
    if not np.array_equal(expected_event,D.event):raise ValueError('Status text and event coding disagree.')
    P=pd.read_csv(probes,dtype=str)
    if P.duplicated(['Gene','ProbeID']).any():raise ValueError('Duplicate probe-map records.')
    mapping={g:P.loc[P.Gene==g,'ProbeID'].tolist() for g in genes}
    if any(not v for v in mapping.values()):raise ValueError('Incomplete six-gene probe mapping.')
    print('RETAINED_INPUT_SAMPLES=',len(D),'OS_EVENTS=',int(D.event.sum()))

# ---------- prespecified univariable continuous Cox ----------
R = []

for g in genes:
    x = D[g].astype(float)
    sd = float(x.std(ddof=1))
    assert np.isfinite(sd) and sd > 0, "STOP zero/invalid SD: " + g

    z = ((x - x.mean()) / sd).to_numpy(dtype=float)

    model = PHReg(
        D["time"].to_numpy(dtype=float),
        z[:,None],
        status=D["event"].to_numpy(dtype=int),
        ties="efron"
    )
    fit = model.fit()

    beta = float(fit.params[0])
    se = float(fit.bse[0])
    p = float(fit.pvalues[0])

    R.append({
        "Gene": g,
        "N_OS": len(D),
        "Events": int(D["event"].sum()),
        "N_GPL96_Probes": len(mapping[g]),
        "MeanLog2MAS5": float(x.mean()),
        "SDLog2MAS5": sd,
        "HR_per_1SD": float(np.exp(beta)),
        "CI95_Low": float(np.exp(beta - 1.96*se)),
        "CI95_High": float(np.exp(beta + 1.96*se)),
        "Cox_P": p
    })

R = pd.DataFrame(R)

R["BH_FDR"] = multipletests(R["Cox_P"], method="fdr_bh")[1]

R["Direction"] = np.where(
    R["HR_per_1SD"] > 1,
    "Higher_expression_worse_OS",
    "Higher_expression_better_OS"
)

R.to_csv(out, index=False, encoding="utf-8-sig")

D.reset_index()[
    ["GSM","Site","OS_status","time","event"] + genes
].to_csv(outx, index=False, encoding="utf-8-sig")

print()
print(R.to_string(index=False))
print()
print("SAVED_COX:", out)
print("SAVED_EXPRESSION:", outx)


pd.DataFrame([{'Gene':g,'ProbeID':probe} for g in genes for probe in mapping[g]]).to_csv(output_dir/'GSE68465_probe_mapping.csv',index=False)
inputs=[f,a] if args.from_raw else [expr,probes]
hashes={}
for path in inputs:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    hashes[path.name]=h.hexdigest()
metadata={'input_mode':'full_geo_inputs' if args.from_raw else 'retained_expression_table','input_sha256':hashes,'patients':len(D),'events':int(D.event.sum()),'model':'Univariable Cox; one continuous gene at a time; Efron ties; sample SD scaling; BH correction across six genes','software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__}}
(output_dir/'recomputation_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
