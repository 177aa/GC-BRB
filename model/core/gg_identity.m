function id = gg_identity(data)
%GG_IDENTITY Conservative leakage-prevention components; no node merging.
N = numel(data.y);
ens = upper(strtrim(erase(string(data.ensembl_ids(:)),char(0))));
sym = upper(strtrim(erase(string(data.gene_symbols(:)),char(0))));
ens(ismissing(ens)) = "";
sym(ismissing(sym)) = "";
ens = regexprep(ens,'\.\d+$','');
if any(ens=="" & sym=="")
    error('GG:Identity','A record has neither ID nor symbol.');
end
edge_a = []; edge_b = [];
for column = 1:2
    if column==1, values=ens; else, values=sym; end
    keep = find(values~="");
    [~,first,g] = unique(values(keep));
    anchor = keep(first(g));
    different = anchor~=keep;
    edge_a = [edge_a;keep(different)]; %#ok<AGROW>
    edge_b = [edge_b;anchor(different)]; %#ok<AGROW>
end
G = graph(edge_a,edge_b,[],N);
group_id = conncomp(G).';
members = accumarray(group_id,(1:N).',[],@(v){v});
K = numel(members);
record_n = zeros(K,1); ens_n=record_n; sym_n=record_n;
missing_n=record_n; labelled_n=record_n; label=nan(K,1);
conflict=false(K,1);
for k = 1:K
    ix=members{k}; record_n(k)=numel(ix);
    ens_n(k)=numel(unique(ens(ix(ens(ix)~=""))));
    sym_n(k)=numel(unique(sym(ix(sym(ix)~=""))));
    missing_n(k)=nnz(ens(ix)=="");
    yy=unique(double(data.y(ix))); yy=yy(isfinite(yy));
    labelled_n(k)=nnz(isfinite(data.y(ix)));
    conflict(k)=numel(yy)>1;
    if numel(yy)==1, label(k)=yy; end
end
id = struct('group_id',group_id,'ensembl',ens,'symbol',sym, ...
    'members',{members},'group_label',label,'label_conflict',conflict);
id.audit = table((1:K).',record_n,ens_n,sym_n,missing_n,labelled_n, ...
    label,conflict,'VariableNames',{'GroupID','RecordN', ...
    'NonemptyEnsemblN','NormalizedSymbolN','MissingEnsemblRecordN', ...
    'LabelledRecordN','Label','LabelConflict'});
id.record_table = table((1:N).',group_id, ...
    string(data.ensembl_ids(:)),string(data.gene_symbols(:)),ens,sym, ...
    double(data.y(:)),'VariableNames',{'GlobalIndex','GroupID', ...
    'EnsemblID','GeneSymbol','EnsemblNormalized','SymbolNormalized','Label'});
end
