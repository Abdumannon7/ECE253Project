%% Batch Nighttime Dehazing with nt_dehaze.p and NIQE/BRISQUE Evaluation
% This script:
% 1. Processes all images in a foggy folder using nt_dehaze.p
% 2. Calculates NIQE and BRISQUE (no-reference quality metrics)
%
% Usage:
%   1. Set foggy_folder to the folder with hazy images
%   2. Set output_folder where dehazed results will be saved
%   3. Run the script

clear all;
close all;

%% Configuration
% Folder containing foggy/hazy images
foggy_folder = '/MATLAB Drive/Project ECE253/results_traditional';

% Output folder for dehazed results
output_folder = './results_matlab2';

% Create output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Get list of images
foggy_files = dir(fullfile(foggy_folder, '*.jpg'));
if isempty(foggy_files)
    foggy_files = [dir(fullfile(foggy_folder, '*.png')); ...
                   dir(fullfile(foggy_folder, '*.bmp'))];
end

num_images = length(foggy_files);
fprintf('Found %d foggy images\n', num_images);

%% Initialize metric storage
niqe_before = zeros(num_images, 1);
niqe_after = zeros(num_images, 1);
brisque_before = zeros(num_images, 1);
brisque_after = zeros(num_images, 1);
valid_count = 0;

%% Process each image
for i = 1:num_images
    fprintf('[%d/%d] Processing: %s\n', i, num_images, foggy_files(i).name);

    try
        % Read foggy image
        foggy_path = fullfile(foggy_folder, foggy_files(i).name);
        foggy_img = imread(foggy_path);
        foggy_img_double = im2double(foggy_img);

        % Calculate NIQE and BRISQUE for original foggy image
        fprintf('  Computing metrics for foggy image...\n');
        niqe_before(i) = niqe(foggy_img);
        brisque_before(i) = brisque(foggy_img);
        fprintf('  Foggy - NIQE: %.4f, BRISQUE: %.4f\n', niqe_before(i), brisque_before(i));

        % Run nt_dehaze.p
        fprintf('  Running nt_dehaze.p...\n');
        dehazed_img_double = nt_dehaze(foggy_img_double);

        % Convert back to uint8 for metrics calculation
        dehazed_img = im2uint8(dehazed_img_double);

        % Calculate NIQE and BRISQUE for dehazed image
        fprintf('  Computing metrics for dehazed image...\n');
        niqe_after(i) = niqe(dehazed_img);
        brisque_after(i) = brisque(dehazed_img);
        fprintf('  Dehazed - NIQE: %.4f, BRISQUE: %.4f\n', niqe_after(i), brisque_after(i));

        % Calculate improvement
        niqe_improvement = niqe_before(i) - niqe_after(i);
        brisque_improvement = brisque_before(i) - brisque_after(i);
        fprintf('  Improvement - NIQE: %.4f, BRISQUE: %.4f\n', niqe_improvement, brisque_improvement);

        % Save dehazed result
        [~, name, ext] = fileparts(foggy_files(i).name);
        output_path = fullfile(output_folder, [name '_dehazed' ext]);
        imwrite(dehazed_img, output_path);
        fprintf('  Saved dehazed image to: %s\n', [name '_dehazed' ext]);

        valid_count = valid_count + 1;

    catch ME
        fprintf('  ✗ Error processing %s: %s\n', foggy_files(i).name, ME.message);
        niqe_before(i) = NaN;
        niqe_after(i) = NaN;
        brisque_before(i) = NaN;
        brisque_after(i) = NaN;
    end

    fprintf('\n');
end

%% Summary Statistics
fprintf('========================================\n');
fprintf('Batch Processing Complete!\n');
fprintf('Total images processed: %d\n', num_images);
fprintf('Valid evaluations: %d\n', valid_count);

if valid_count > 0
    % Remove NaN values
    valid_niqe_before = niqe_before(~isnan(niqe_before));
    valid_niqe_after = niqe_after(~isnan(niqe_after));
    valid_brisque_before = brisque_before(~isnan(brisque_before));
    valid_brisque_after = brisque_after(~isnan(brisque_after));

    fprintf('\n========================================\n');
    fprintf('NIQE Statistics (Lower = Better):\n');
    fprintf('----------------------------------------\n');
    fprintf('  Before (Foggy):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_niqe_before));
    fprintf('    Std:    %.4f\n', std(valid_niqe_before));
    fprintf('    Median: %.4f\n', median(valid_niqe_before));
    fprintf('\n');
    fprintf('  After (Dehazed):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_niqe_after));
    fprintf('    Std:    %.4f\n', std(valid_niqe_after));
    fprintf('    Median: %.4f\n', median(valid_niqe_after));
    fprintf('\n');
    fprintf('  Improvement (Before - After):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_niqe_before) - mean(valid_niqe_after));
    fprintf('    Median: %.4f\n', median(valid_niqe_before) - median(valid_niqe_after));
    fprintf('\n');

    fprintf('========================================\n');
    fprintf('BRISQUE Statistics (Lower = Better):\n');
    fprintf('----------------------------------------\n');
    fprintf('  Before (Foggy):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_brisque_before));
    fprintf('    Std:    %.4f\n', std(valid_brisque_before));
    fprintf('    Median: %.4f\n', median(valid_brisque_before));
    fprintf('\n');
    fprintf('  After (Dehazed):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_brisque_after));
    fprintf('    Std:    %.4f\n', std(valid_brisque_after));
    fprintf('    Median: %.4f\n', median(valid_brisque_after));
    fprintf('\n');
    fprintf('  Improvement (Before - After):\n');
    fprintf('    Mean:   %.4f\n', mean(valid_brisque_before) - mean(valid_brisque_after));
    fprintf('    Median: %.4f\n', median(valid_brisque_before) - median(valid_brisque_after));
    fprintf('\n');

    fprintf('========================================\n');
    fprintf('Results saved to: %s\n', output_folder);
    fprintf('========================================\n');

    %% Save results to CSV
    results_table = table(string({foggy_files.name}'), ...
                          niqe_before, niqe_after, ...
                          brisque_before, brisque_after, ...
                          'VariableNames', {'Filename', 'NIQE_Before', 'NIQE_After', 'BRISQUE_Before', 'BRISQUE_After'});
    csv_path = fullfile(output_folder, 'niqe_brisque_results.csv');
    writetable(results_table, csv_path);
    fprintf('Detailed results saved to: %s\n', csv_path);

    %% Save summary statistics
    summary_table = table({'NIQE_Before'; 'NIQE_After'; 'NIQE_Improvement'; ...
                           'BRISQUE_Before'; 'BRISQUE_After'; 'BRISQUE_Improvement'}, ...
                          [mean(valid_niqe_before); mean(valid_niqe_after); mean(valid_niqe_before) - mean(valid_niqe_after); ...
                           mean(valid_brisque_before); mean(valid_brisque_after); mean(valid_brisque_before) - mean(valid_brisque_after)], ...
                          [median(valid_niqe_before); median(valid_niqe_after); median(valid_niqe_before) - median(valid_niqe_after); ...
                           median(valid_brisque_before); median(valid_brisque_after); median(valid_brisque_before) - median(valid_brisque_after)], ...
                          'VariableNames', {'Metric', 'Mean', 'Median'});
    summary_csv_path = fullfile(output_folder, 'summary_statistics.csv');
    writetable(summary_table, summary_csv_path);
    fprintf('Summary statistics saved to: %s\n', summary_csv_path);
end

fprintf('\n========================================\n');
fprintf('Note: Lower NIQE and BRISQUE scores indicate better image quality.\n');
fprintf('Positive improvement values mean the dehazed image is better.\n');
fprintf('========================================\n');
