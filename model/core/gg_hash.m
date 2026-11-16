function hex = gg_hash(value,is_file)
%GG_HASH SHA-256 of UTF-8 text or the exact bytes of a file.
if nargin < 2, is_file = false; end
md = java.security.MessageDigest.getInstance('SHA-256');
if is_file
    fid = fopen(value,'rb');
    if fid < 0, error('GG:Read','Cannot read %s',value); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    while true
        block = fread(fid,1048576,'*uint8');
        if isempty(block), break; end
        md.update(typecast(block(:),'int8'));
    end
else
    bytes = unicode2native(char(value),'UTF-8');
    md.update(typecast(uint8(bytes(:)),'int8'));
end
bytes = typecast(md.digest(),'uint8');
hex = lower(reshape(dec2hex(bytes,2).',1,[]));
end
