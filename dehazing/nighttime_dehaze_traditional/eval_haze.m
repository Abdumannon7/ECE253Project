original = imread('data/flickr1.bmp');
dehazed = imread('e/img.png');

% Calculate metrics for both
niqe_original = niqe(original);
niqe_dehazed = niqe(dehazed);

brisque_original = brisque(original);
brisque_dehazed = brisque(dehazed);

% Display comparison
fprintf('--- Original Image ---\n');
fprintf('NIQE: %.4f\n', niqe_original);
fprintf('BRISQUE: %.4f\n', brisque_original);

fprintf('\n--- Dehazed Image ---\n');
fprintf('NIQE: %.4f\n', niqe_dehazed);
fprintf('BRISQUE: %.4f\n', brisque_dehazed);

fprintf('\n--- Improvement ---\n');
fprintf('NIQE Improvement: %.4f (negative is better)\n', niqe_dehazed - niqe_original);
fprintf('BRISQUE Improvement: %.4f (negative is better)\n', brisque_dehazed - brisque_original);

% Ensure output is displayed when calling eval_haze
disp('Metrics calculated and displayed successfully.');