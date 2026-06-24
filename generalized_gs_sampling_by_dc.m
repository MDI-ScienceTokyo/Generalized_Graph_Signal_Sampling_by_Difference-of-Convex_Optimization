function [x_rec, S] = generalized_gs_sampling_by_dc(x_true, P, N, M, param)
% 
% [Description]
%   Computes the optimal graph sampling operator S via Difference-of-Convex (DC) 
%   optimization and performs graph signal recovery following the framework
%   proposed in the companion paper.
%
% [Inputs]
%   x_true : Ground-truth graph signal (N x 1)
%   P      : Condition/Prior-dependent matrix (Refer to Table IV in the paper)
%   N      : Number of vertices in the graph
%   M      : Size of the sampled signal (M x 1)
%   param  : Parameter structure containing:
%       .design     - 1: Design (i),  2: Design (ii),  3: Design (iii)
%       .gs_prior   - 'Subspace', 'Smoothness', or 'Stochastic'
%       .gamma1, .gamma2 - Step sizes for the DC optimization loop
%       .epsi       - Norm upper bound for Design (i)
%       .lambda     - Regularization weight for Design (ii) and (iii)
%       .mina, .maxb - Lower and upper bounds for the box constraints
%       (Prior-specific matrices)
%       .A          - Subspace Prior: Generator matrix
%       .F          - Smoothness Prior: Smoothness operator
%       .Gamma_x    - Stochastic Prior: Autocorrelation matrix of the signal
%
% [Outputs]
%   x_rec  : Recovered graph signal (N x 1)
%   S      : Designed optimal sampling operator (N x M)

    % =========================================================================
    % 1. Sampling Operator Design (DC Optimization Loop)
    % =========================================================================
    MAX_ITERATIONS = 300000;    % Maximum number of iterations
    EQUAL_THRESHOLD = 1e-5;     % Convergence tolerance threshold
    
    gamma1 = param.gamma1;
    gamma2 = param.gamma2;
    
    % Initialize variables (Fixed seed for strict reproducibility)
    rng(3000); 
    s_t = randn(N, M);
    z_t = P * s_t;
    
    for t = 1:MAX_ITERATIONS
        % Step 1: Update S^{(t+1)}
        grad_input = s_t + gamma1 * P' * z_t;
        
        if param.design == 1
            % Design (i): Projection onto the L2-ball centered at f = 0
            s_t1 = Proj_L2ball(grad_input, 0, param.epsi);
        elseif param.design == 2
            % Design (ii): Squared Frobenius norm regularizer + Box constraints
            s_t1 = prox_FNandBoxConst(grad_input, gamma1, param.lambda, param.mina, param.maxb);
        elseif param.design == 3
            % Design (iii): L1-norm regularizer + Box constraints
            s_t1 = prox_L1andBoxConst(grad_input, gamma1, param.lambda, param.mina, param.maxb);
        else
            error('Invalid design mode. Please select 1, 2, or 3.');
        end

        % Step 2: Update Z^{(t+1)}
        zdash = z_t + gamma2 * P * s_t1;
        % Apply the conjugate operator (Proximal operator of the nuclear norm)
        z_t1 = zdash - gamma2 * prox_NN(zdash / gamma2, 1 / gamma2);
        
        % Step 3: Convergence Check
        diff_val = norm(s_t1 - s_t, 'fro') / norm(s_t, 'fro');
        if diff_val < EQUAL_THRESHOLD
            s_t = s_t1;
            break;
        end
        s_t = s_t1;
        z_t = z_t1;
    end
    S = s_t;

    % =========================================================================
    % 2. Sampling Process
    % =========================================================================
    c = S' * x_true; % Sampled signal

    % =========================================================================
    % 3. Graph Signal Recovery (Constructing W and H based on Table III)
    % =========================================================================
    switch param.gs_prior
        case 'Subspace'
            % Subspace Prior (Unconstrained)
            W = param.A;
            H = pinv(S' * param.A);
            
        case 'Smoothness'
            % Smoothness Prior (Unconstrained)
            F = param.F;
            W = pinv(F' * F) * S;
            H = pinv(S' * W);
            
        case 'Stochastic'
            % Stochastic Prior (Unconstrained, Gamma_eta = 0 for noiseless)
            W = param.Gamma_x * S;
            H = pinv(S' * param.Gamma_x * S);
            
        otherwise
            error('Unknown prior type specified.');
    end

    % Compute the final recovered graph signal
    x_rec = W * H * c;
end

% --- Local Functions (Mathematical Projections & Proximal Operators) ---

function result = Proj_L2ball(X, f, epsilon)
    % Projection onto the L2-ball centered at f with radius epsilon
    temp = X - f;
    radius = norm(temp(:), 2);
    result = X;
    if radius > epsilon
        result = f + (epsilon / radius) * temp;
    end
end

function result = prox_NN(X, lambda)
    % Proximal operator of the nuclear norm via economic SVD
    [U, S, V] = svd(X, 'econ');
    S_ST = max(S - lambda, 0);
    result = U * S_ST * V';
end

function result = prox_FNandBoxConst(X, gamma1, lambda, a, b)
    % Proximal operator for the squared Frobenius norm with box constraints [a, b]
    result = X / (2 * lambda * gamma1 + 1);
    result(result < a) = a;
    result(result > b) = b;
end

function result = prox_L1andBoxConst(X, gamma1, lambda, a, b)
    % Proximal operator for the L1-norm (soft-thresholding) with box constraints [a, b]
    thres = gamma1 * lambda;
    X_soft = max(abs(X) - thres, 0) .* sign(X);
    result = max(a, min(b, X_soft));
end