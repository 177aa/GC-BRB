function [data,id,base] = gg_load_data(cfg)
if ~usejava('jvm'),error('GG:JVM','MATLAB JVM is required for SHA-256.');end
if ~isfile(cfg.prepared_path)
    error('GG:Source',['Prepared data not found. Edit source_project_root ' ...
        'in gg_config.m. Expected: %s'],cfg.prepared_path);
end
if ~strcmp(cfg.stage,'preflight') && isempty(which('fitclinear'))
    error('GG:Toolbox','Statistics and Machine Learning Toolbox is required.');
end
if ~isfolder(cfg.result_root),mkdir(cfg.result_root);end
S=load(cfg.prepared_path,'data');
if ~isfield(S,'data'),error('GG:Source','Missing data variable.');end
data=S.data;
required={'y','X','global_keys','ensembl_ids','gene_symbols', ...
    'layers','layer_names','num_global_genes','num_layers','presence'};
for j=1:numel(required)
    if ~isfield(data,required{j}),error('GG:Source','Missing %s.',required{j});end
end
if size(data.X,2)~=64 || data.num_layers~=6 ...
        || size(data.X,1)~=numel(data.y) || any(~isfinite(data.X(:)))
    error('GG:Source','Unexpected omics dimensions or nonfinite values.');
end
ref=jsondecode(fileread(fullfile(cfg.project_root,'reference','expected_identity.json')));
yy=repmat("NA",numel(data.y),1);yy(data.y==0)="0";yy(data.y==1)="1";
text=join(string(data.ensembl_ids(:))+sprintf('\t')+ ...
    string(data.gene_symbols(:))+sprintf('\t')+yy,sprintf('\n'));
identity_hash=gg_hash(text);
if ~strcmp(identity_hash,ref.IdentitySHA256)
    error('GG:Mapping','Current record order/identity/labels differ from reference.');
end
id=gg_identity(data);
writetable(id.audit,fullfile(cfg.result_root,'identity_group_audit.csv'));
writetable(id.record_table,fullfile(cfg.result_root,'identity_record_map.csv'));
writetable(id.audit(id.audit.NonemptyEnsemblN>1,:), ...
    fullfile(cfg.result_root,'ambiguous_multiensembl_groups.csv'));
if any(id.label_conflict)
    error('GG:Conflict','Conflicting labels; inspect identity_group_audit.csv.');
end
if height(id.audit)~=ref.GroupN
    error('GG:Mapping','Identity group count differs from independent reference.');
end
source_files=[dir(fullfile(cfg.project_root,'*.m')); ...
    dir(fullfile(cfg.project_root,'core','*.m')); ...
    dir(fullfile(cfg.project_root,'legacy_core','*.m')); ...
    dir(fullfile(cfg.project_root,'tests','*.m'))];
names=string(fullfile({source_files.folder},{source_files.name}));
[names,order]=sort(names);source_files=source_files(order);
parts=cell(numel(source_files),1);
for j=1:numel(source_files)
    parts{j}=[char(erase(names(j),[cfg.project_root filesep])) ':' ...
        gg_hash(char(names(j)),true)];
end
manifest=struct('Protocol',cfg.protocol_version,'Stage',cfg.stage,'Profile',cfg.profile, ...
    'PreparedPath',cfg.prepared_path,'PreparedSHA256',gg_hash(cfg.prepared_path,true), ...
    'IdentitySHA256',identity_hash,'CodeSHA256',gg_hash(strjoin(parts,'|')), ...
    'MATLABRelease',version('-release'),'Config',cfg);
scientific=cfg;
scientific=rmfield(scientific,{'stage','target_datasets','num_runs','resume'});
fingerprint=rmfield(manifest,{'Stage','Config'});fingerprint.Config=scientific;
base=gg_hash(jsonencode(fingerprint));manifest.BaseFingerprint=base;
path=fullfile(cfg.result_root,'protocol.json');
if isfile(path)
    old=jsondecode(fileread(path));
    if ~isfield(old,'BaseFingerprint') || ~strcmp(old.BaseFingerprint,base)
        error('GG:Protocol',['Existing output has another fingerprint. ' ...
            'Use a new result directory; do not mix results.']);
    end
end
gg_jsonwrite(path,manifest);
end
