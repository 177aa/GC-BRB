function gg_save(path,payload)
%GG_SAVE Publish a complete checkpoint only after save has succeeded.
folder = fileparts(path);
if ~isfolder(folder), mkdir(folder); end
tmp = [tempname(folder) '.mat'];
cleanup = onCleanup(@() remove_temp(tmp)); %#ok<NASGU>
save(tmp,'-struct','payload','-v7.3');
[ok,msg] = movefile(tmp,path,'f');
if ~ok, error('GG:Save','%s',msg); end
end
function remove_temp(path)
if isfile(path), delete(path); end
end
