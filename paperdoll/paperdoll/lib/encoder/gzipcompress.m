function output = gzipcompress(input)
%GZIPCOMPRESS Compress input bytes with GZIP.
%
%    output = gzipcompress(input)
%
% The function takes a char, int8, or uint8 array INPUT and returns
% compressed bytes OUTPUT as a uint8 array. Note that the compression
% doesn't preserve input dimensions. JAVA must be enabled to use the
% function.
%
% See also gzipdecompress typecast

error(nargchk(1, 1, nargin));
error(javachk('jvm'));
if ischar(input), input = uint8(input); end
if ~isa(input, 'int8') && ~isa(input, 'uint8')
    error('Input must be either char, int8 or uint8.');
end

buffer = java.io.ByteArrayOutputStream();
gzip = java.util.zip.GZIPOutputStream(buffer);
gzip.write(input, 0, numel(input));
gzip.close();
output = typecast(buffer.toByteArray(), 'uint8')';

end

