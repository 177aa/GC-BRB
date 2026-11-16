function archive=gg_make_return_pack()
%GG_MAKE_RETURN_PACK Return logs/predictions/plans, not graph/model caches.
root=fileparts(fileparts(mfilename('fullpath')));
result_root=fullfile(root,'results');
if ~isfolder(result_root),error('GG:Return','No results directory yet.');end
tmp=tempname;mkdir(tmp);cleanup=onCleanup(@()rmdir(tmp,'s')); %#ok<NASGU>
files=dir(fullfile(result_root,'**','*'));count=0;
for k=1:numel(files)
    if files(k).isdir,continue;end
    name=files(k).name;[~,~,extension]=fileparts(name);
    if ~ismember(lower(string(extension)),[".csv",".json",".txt",".mat"]),continue;end
    if strcmp(extension,'.mat') && ~any(strcmp(name,{'plan.mat','run_result.mat'})),continue;end
    if any(strcmp(name,{'baseline_features.csv','candidate_entry_scores.csv'})),continue;end
    source=fullfile(files(k).folder,name);
    relative=extractAfter(string(source),strlength(string(root))+1);
    destination=fullfile(tmp,char(relative));
    if ~isfolder(fileparts(destination)),mkdir(fileparts(destination));end
    copyfile(source,destination);count=count+1;
end
if count==0,error('GG:Return','No diagnostic files found.');end
archive=fullfile(root,['GC_BRB_GROUPED_RETURN_' datestr(now,'yyyymmdd_HHMMSS') '.zip']);
zip(archive,{'results'},tmp);
fprintf('Return package created (%d diagnostic files):\n%s\n',count,archive);
end
