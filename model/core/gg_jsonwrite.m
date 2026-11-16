function gg_jsonwrite(path,value)
text = jsonencode(value,'PrettyPrint',true);
fid = fopen(path,'w','n','UTF-8');
if fid < 0, error('GG:Write','Cannot write %s',path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',text);
end
