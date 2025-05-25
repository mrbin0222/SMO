function [y,var] = awgn(sig,reqSNR,varargin)
%AWGN Add white Gaussian noise to a signal.
%   Y = AWGN(X,SNR) adds white Gaussian noise to X. The SNR is in dB.
%   The power of X is assumed to be 0 dBW. If X is complex, then
%   AWGN adds complex noise.
%
%   Y = AWGN(X,SNR,SIGPOWER) when SIGPOWER is numeric, it represents
%   the signal power in dBW. When SIGPOWER is 'measured', AWGN measures
%   the signal power before adding noise.
%
%   Y = AWGN(X,SNR,SIGPOWER,S) uses S to generate random noise samples with
%   the RANDN function. S can be a random number stream specified by
%   RandStream. S can also be an integer, which seeds a random number
%   stream inside the AWGN function. If you want to generate repeatable
%   noise samples, then either reset the random stream input before calling
%   AWGN or use the same seed input.
%
%   Y = AWGN(..., POWERTYPE) specifies the units of SNR and SIGPOWER. To
%   specify POWERTYPE, both SNR and SIGPOWER must be specified. POWERTYPE
%   can be 'db' or 'linear'. If POWERTYPE is 'db', then SNR is measured in
%   dB and SIGPOWER is measured in dBW.  If POWERTYPE is 'linear', then SNR
%   is measured as a ratio and SIGPOWER is measured in watts assuming 1 ohm
%   reference load.
%
%   [Y,VAR] = AWGN(...) returns the total noise variance used to generate
%   random noise samples.
%
%   Class Support
%   -------------
%   Input signal, X, can be a numeric array, dlarray, or gpuArray
%   of underlying type double or single. Output signal, Y, is a dlarray if 
%   X is a dlarray. Output signal, Y, is a gpuArray if X is a gpuArray.
%
%   Notes
%   -----
%   1. X can be an array of up to three dimensions specified as
%       Ns-by-Nc-by-Nb, where Ns is the number of samples, Nc is the number
%       of channels, and Nb is the number of batches.
%   2. If X is a 3D array, then SNR can either be a scalar or a vector of length Nb. If SNR is
%       a scalar, then the same SNR value is applied to all batches.
%   3. Specifying a random number stream or seed, S, is not supported if X is a dlarray or gpuArray.
%   4. Specifying SIGPOWER as 'measured' is not supported if X is a dlarray
%   or gpuArray, or if SNR is a vector.
%
%   Example 1:
%        % To specify the power of X to be 0 dBW and add noise to produce
%        % an SNR of 10dB, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0);
%
%   Example 2:
%        % To specify the power of X to be 3 watts and add noise to
%        % produce a linear SNR of 4, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,4,3,'linear');
%
%   Example 3: 
%        % To cause AWGN to measure the power of X and add noise to
%        % produce a linear SNR of 4, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,4,'measured','linear');
%
%   Example 4:
%        % To specify the power of X to be 0 dBW, add noise to produce
%        % an SNR of 10dB, and utilize a local random stream, use:
%        S = RandStream('mt19937ar','Seed',5489);
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0,S);
%
%   Example 5:
%        % To specify the power of X to be 0 dBW, add noise to produce
%        % an SNR of 10dB, and produce reproducible results, use:
%        rng('default')
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0);
%
%
%   See also convertSNR, WGN, BSC, RANDN, RandStream/RANDN.

%   Copyright 1996-2023 The MathWorks, Inc.

%#codegen

narginchk(2,5);

% Validate signal input
validateattributes(sig, {'numeric'}, ...
    {'nonempty'}, 'awgn', 'signal input');

dlarraySig = isa(sig,'dlarray');
gpuArraySig = isa(sig,'gpuArray');

% Formatted dlarrays are not supported
if dlarraySig && ~isempty(dims(sig))
    error(message('comm:checkinp:InvalidDlarrayFormat'));
end

numBatches = size(sig,3);

% Validate SNR input
validateattributes(reqSNR, {'numeric'}, ...
    {'real','vector','nonempty'}, 'awgn', 'SNR input');

vectorSNR = ~isscalar(reqSNR);

% Dlarray/gpuArray SIG do NOT support ndims(sig) > 3
if (dlarraySig || gpuArraySig || vectorSNR) & ndims(sig)>3
    error(message('comm:awgn:InvalidInputSize'));
end

if vectorSNR && length(reqSNR)~=numBatches
    error(message('comm:awgn:InvalidSNRSize',numBatches));
end

% Validate signal power
if nargin >= 3
    if strcmpi(varargin{1}, 'measured')
        if vectorSNR || dlarraySig || gpuArraySig
            error(message('comm:awgn:UnsupportedMeasuredSyntax'));
        end
        sigPower = sum(abs(sig(:)).^2)/numel(sig); % linear
    else
        validateattributes(varargin{1}, {'numeric'}, ...
            {'real','scalar','nonempty'}, 'awgn', 'signal power input');
        sigPower = varargin{1}; % linear or dB
    end
else
    sigPower = 1; % linear, default
end

% Validate state or power type
if nargin >= 4
    if comm.internal.utilities.isCharOrStringScalar(varargin{2}) && ...
            all(~strcmpi(varargin{2}, {'db','linear'}))
        error(message('comm:awgn:InvalidPowerType'));
    end

    isStream = ~isempty(varargin{2}) && ~comm.internal.utilities.isCharOrStringScalar(varargin{2});
    if isStream && (gpuArraySig||dlarraySig)
        error(message('comm:awgn:UnsupportedSeedStreamSyntax'));
    end

    if isStream && ~isa(varargin{2}, 'RandStream') % Random stream seed
        validateattributes(varargin{2}, {'double'}, ...
            {'real','scalar','nonnegative','integer','<',2^32}, ...
            'awgn', 'seed input');
    end
else % Use default stream & seed
    isStream = false;
end

% Validate power type
if nargin == 5
    if comm.internal.utilities.isCharOrStringScalar(varargin{2}) % Type has been specified as the 4th input
        error(message('comm:awgn:InputAfterPowerType'));
    end
    if all(~strcmpi(varargin{3}, {'db','linear'}))
        error(message('comm:awgn:InvalidPowerType'));
    end
end

isLinearScale = ((nargin == 4) && ~isStream && strcmpi(varargin{2}, 'linear')) || ...
    ((nargin == 5) && strcmpi(varargin{3}, 'linear'));

% Cross-validation
if isLinearScale && (sigPower < 0)
    error(message('comm:awgn:InvalidSigPowerForLinearMode'));
end

if isLinearScale && any(reqSNR < 0)
    error(message('comm:awgn:InvalidSNRForLinearMode'));
end

if ~isLinearScale  % Convert signal power and SNR to linear scale
    if (nargin >= 3) && ~comm.internal.utilities.isCharOrStringScalar(varargin{1}) % User-specified signal power
        sigPower = 10^(sigPower/10);
    end
    reqSNR = 10.^(reqSNR./10);
end

if vectorSNR
    noisePower = reshape(sigPower./reqSNR,1,1,numBatches);
else
    noisePower = sigPower./reqSNR;
end

if isStream
    if isa(varargin{2}, 'RandStream')
        stream = varargin{2};
    elseif isempty(coder.target)
        stream = RandStream('mt19937ar', 'Seed', varargin{2});
    else
        stream = coder.internal.RandStream('mt19937ar', 'Seed', varargin{2});
    end
    y = sig + sqrt(noisePower).*randn(stream,size(sig),"like",sig);
else
    y = sig + sqrt(noisePower).*randn(size(sig),"like",sig);
end

if nargout == 2
    var = reshape(noisePower, size(reqSNR));
end
