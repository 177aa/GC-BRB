function root = gg_setup()
%GG_SETUP Use this project's numerical functions before any older project.
root = fileparts(mfilename('fullpath'));
addpath(root,'-begin');
addpath(fullfile(root,'legacy_core'),'-begin');
addpath(fullfile(root,'core'),'-begin');
addpath(fullfile(root,'tests'),'-begin');
names = {'cmaes','fit_omics_evidence','multilayer_brb_forward_v3layer'};
for k = 1:numel(names)
    if ~startsWith(which(names{k}),[root filesep])
        error('GG:Path','Unexpected function on MATLAB path: %s',names{k});
    end
end
end
