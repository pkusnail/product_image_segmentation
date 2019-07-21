function samples = apply( config, samples, varargin )
%APPLY Apply feature transform.

  assert(isstruct(config));
  assert(isstruct(samples));

  % Get options.
  FORCE = false;
  ENCODE = false;
  for i = 1:2:numel(varargin)
    switch varargin{i}
      case 'Force', FORCE = varargin{i+1};
      case 'Encode', ENCODE = varargin{i+1};
    end
  end
  
  % Quit if it's already there.
  if ~FORCE && isfield(samples, config.output)
    return
  end
  
  % Resolve dependency.
  assert(isfield(samples, config.input));
  
  % Compute mr8 feature.
  [samples.(config.output)] = deal([]);
  for i = 1:numel(samples)
    sample = feature_calculator.decode(samples(i), config.input);
    rgb_image = imread_or_decode(sample.(config.input), 'jpg');
    if size(rgb_image, 3) == 1, rgb_image = repmat(rgb_image, [1,1,3]); end
    samples(i).(config.output) = vl_xyz2lab(vl_rgb2xyz(rgb_image));
    if ENCODE
      samples(i) = feature_calculator.encode(samples(i), config.output);
    end
  end

end