function snr_db = compute_row_snr(idealOut, out_eq)
%COMPUTE_ROW_SNR Row-wise SNR between idealOut and out_eq (4 rows expected).
%   snr_db = compute_row_snr(idealOut, out_eq) returns a 4x1 vector where each
%   entry is 10*log10(signal_power/noise_power) for the corresponding row.

if ~isequal(size(idealOut), size(out_eq))
    error('idealOut and out_eq must have the same size.');
end

idealOut = double(idealOut);
out_eq   = double(out_eq);

diff = out_eq - idealOut;
snr_db = zeros(4, 1);
for r = 1:4
    sig_power = mean(abs(idealOut(r, :)).^2);
    noise_power = mean(abs(diff(r, :)).^2);
    snr_db(r) = 10*log10(sig_power / max(noise_power, eps));
end
end
