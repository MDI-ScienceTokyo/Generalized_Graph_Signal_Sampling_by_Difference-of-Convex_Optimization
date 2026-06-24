% =========================================================================
% DEMO: Generalized Graph Signal Sampling by Difference-of-Convex Optimization
% =========================================================================
%
% [Description]
%   This script is the demonstration program for the algorithm proposed in 
%   the companion paper (see Citation below). It implements the novel 
%   generalized graph signal sampling operator design via Difference-of-
%   Convex (DC) optimization and executes the signal recovery framework.
%
% [Reference / Citation]
%   If you use this code or find the method helpful in your research, please 
%   cite the following paper:
%
%   K. Yamashita, K. Naganuma, and S. Ono, "Generalized Graph Signal Sampling 
%   by Difference-of-Convex Optimization," arXiv preprint arXiv:2306.14634, 2025.
%   URL: https://arxiv.org/abs/2306.14634
%
% [Prerequisites]
%   - MATLAB (R2022b or later recommended)
%   - GSPBox: Graph Signal Processing Toolbox
%     Official Website: https://epfl-lts2.github.io/gspbox-html/
%   - generalized_gs_sampling_by_dc.m (Must be in the same directory or path)
%
% [License]
%   Copyright (c) 2026 Keitaro Yamashita (MDI Lab, Institute of Science Tokyo).
%   All rights reserved.
%   Licensed under the MIT License. (See LICENSE file in the repository root)
%
% =========================================================================

clear; clc; close all;

% =========================================================================
% 1. Experimental Conditions & Parameter Settings
% =========================================================================
% --- Prior Information Choice ---
% 'BL' (Subspace prior), 'PWL' (Smoothness prior), 'SGS' (Stochastic prior)
signal_type = 'BL'; 

% --- Sampling Design Choice (Array input allowed for multiple designs) ---
% 1: Design (i), 2: Design (ii), 3: Design (iii)
design_choices = [1,2,3]; 

% --- Fundamental Parameters ---
N = 256;            % Number of vertices in the graph
M = 16;             % Number of samples

% --- Optimization Parameters ---
param.gamma1 = 0.001;
param.gamma2 = 0.001;
param.mina = 0;
param.maxb = 1;

% Design-Specific Regularization/Constraints
param.epsi       = sqrt(N * M) / 4; % For Design (i): Upper bound of the Frobenius norm
param.lambda_ii  = 0.5;             % For Design (ii): Regularization parameter
param.lambda_iii = 0.1;             % For Design (iii): Regularization parameter


% =========================================================================
% 2. Graph Generation, Graph-Dependent Parameters, & Signal Generation
% =========================================================================
fprintf('--- Generating Graph ---\n');
% (2-a) Generate the graph structure and compute eigenvalues
G = gsp_sensor(N);
G = gsp_compute_fourier_basis(G);

% (2-b) User-defined parameters configured after graph generation
switch signal_type
    case 'BL'
        param.BL_K = 16;                    % Subspace dimension (K)
        
    case 'PWL'
        param.PWL_density = 8 / N;          % Sparse density for signal generation
        % Define diagonal components of F using graph eigenvalues (G.e, G.lmax)
        param.F_diag = G.e / G.lmax + 0.1;  
        
    case 'SGS'
        % Define Gamma_x_hat using graph eigenvalues (G.e, G.lmax)
        param.Gamma_x_hat = exp(-((2 * G.e - G.lmax) / sqrt(G.lmax)).^2);
end

% (2-c) Generate the true graph signal and the measurement condition matrix P
fprintf('--- Generating Signal (%s) ---\n', signal_type);
[x_true, P, param] = generate_signal_from_prior(signal_type, G, param);

num_plots = length(design_choices) + 1;
fig_width = min(350 * num_plots, 1800); 
fig_handle = figure('Name', sprintf('Signal Recovery Comparison (%s Prior)', ...
                    param.gs_prior), 'Position', [100, 100, fig_width, 400]);
tiled_handle = tiledlayout(1, num_plots, 'TileSpacing', 'compact', 'Padding', 'compact');
c_limits = setup_figure_and_plot_original(G, x_true, tiled_handle);

% =========================================================================
% 3. Algorithm Execution & Evaluation
% =========================================================================
roman_labels = {'(i)', '(ii)', '(iii)'};

for i = 1:length(design_choices)
    current_design = design_choices(i);
    param.design = current_design;
    
    % Set regularizer weight lambda based on the current design
    if current_design == 2
        param.lambda = param.lambda_ii;
    elseif current_design == 3
        param.lambda = param.lambda_iii;
    else
        param.lambda = 0; % Not used in Design (i)
    end
    
    design_str = roman_labels{current_design};

    fprintf('\n--- Executing Optimization (Design %s) ---\n', design_str);
    
    % Design the sampling operator and recover the signal
    tic;
    [x_rec, S] = generalized_gs_sampling_by_dc(x_true, P, N, M, param);
    elapsed_time = toc;
    
    % Evaluate performance and plot the recovered signal
    evaluate_and_plot_result(x_true, x_rec, G, design_str, elapsed_time, c_limits, tiled_handle);
end

fprintf('\n=== All processes completed successfully ===\n');


% =========================================================================
% Local Functions (Encapsulated Background Processes)
% =========================================================================

function [x_true, P, param] = generate_signal_from_prior(signal_type, G, param)
    % Generates the ground-truth graph signal and the matrix P based on the chosen prior.
    rng(1000);
    N = G.N;

    switch signal_type
        case 'BL'
            param.gs_prior = 'Subspace';
            K = param.BL_K; 
            
            param.A = G.U(:, 1:K);
            d = randn(K, 1) + 1;
            x_true = param.A * d;           
            P = param.A';
            
        case 'PWL'
            param.gs_prior = 'Smoothness';
            density = param.PWL_density;
            
            b = 2 * sprand(N, 1, density);
            Lambda = find(b ~= 0);
            b(Lambda) = b(Lambda) - 1;
            pinvL = pinv(full(G.L));
            x_true = pinvL * b;
            x_true = x_true / max(abs(x_true)) * 0.5;
            
            % Construct the smoothness operator F from the pre-computed diagonal
            param.F = G.U * diag(param.F_diag) * G.U';
            
            [~, Sigma_F, V_F] = svd(param.F);
            P = pinv(Sigma_F) * V_F';
            
        case 'SGS'
            param.gs_prior = 'Stochastic';
            
            % Fetch the pre-computed Gamma_x_hat
            Gamma_x_hat = param.Gamma_x_hat;
            
            param.Gamma_x = G.U * diag(Gamma_x_hat) * G.U';
            x_true = mvnrnd(zeros(N, 1), param.Gamma_x, 1).'; 
            P = diag(sqrt(Gamma_x_hat)) * G.U';
            
        otherwise
            error('Invalid prior choice. Please select a valid prior.');
    end
end

function c_limits = setup_figure_and_plot_original(G, x_true, tiled_handle)
    % Move to the first tile and plot the original ground-truth signal
    nexttile(tiled_handle, 1);
    gsp_plot_signal(G, x_true);
    title('Original Signal', 'Interpreter', 'tex');
    colorbar;
    axis square;
    axis off;
    c_limits = clim; 
end

function evaluate_and_plot_result(x_true, x_rec, G, design_str, elapsed_time, c_limits, tiled_handle)
    % Calculates metrics and outputs quantitative metrics to terminal
    N = length(x_true);
    mse = norm(x_true - x_rec)^2 / N;
    mse_dB = db(mse);
    
    fprintf('Execution Time : %.2f seconds\n', elapsed_time);
    fprintf('Recovery MSE   : %.4e (%.2f dB)\n', mse, mse_dB);
    
    % Move to the next available tile and plot the recovered signal
    nexttile(tiled_handle);
    gsp_plot_signal(G, x_rec);
    title(sprintf('Design %s\nMSE: %.2f dB', design_str, mse_dB));
    clim(c_limits); 
    colorbar;
    axis square;
    axis off;
end