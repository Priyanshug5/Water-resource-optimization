clc; clear; close all;

%% Simplified Water Resource Optimization for Karnataka–Tamil Nadu Basin

% System Parameters
n_months    = 12;
months      = 1:n_months;
months_lbl  = {'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};

% Seasonal Agri values (crores per BCM)
agri_USD_K = [12,12,8,8,5,15,15,15,15,8,8,12] * 1e3 * 86;    % K harvest pattern
agri_USD_T = [13,14,14,12,8,5,5,8,12,15,15,15] * 1e3 * 86;    % TN harvest pattern

% Industrial and transfer profits
ind_USD      = [190e3,190e3] * 86;     % (K, TN)
transfer_USD = 200e3 * 86;
storage_USD  = [2e3,2e3] * 86;
penalty_agri = 500e3 * 86;
penalty_ind  = 100e3 * 86;

economic_params = struct( ...
    'agri_K', agri_USD_K, 'agri_T', agri_USD_T, ...
    'ind',    ind_USD,    'transfer', transfer_USD, ...
    'storage', storage_USD, 'penalty_agri', penalty_agri, 'penalty_ind', penalty_ind ...
);
region_weights = [0.6, 0.4];      % weights for (K, TN)
env_flow       = 0.05;            % BCM/month
sla            = struct('ind_min',0.8, 'agri_min',0.7);

% Household minimum releases (BCM/month)
hh_min_K = 0.12 * ones(1,n_months);
hh_min_T = 0.10 * ones(1,n_months);

% Reservoir parameters
reservoirs.K = init_reservoir_params('Karnataka', 3.244283, 2.094905);
reservoirs.T = init_reservoir_params('Tamil Nadu', 3.646771, 1.544823);

% Decision variables
var_names = {'K_ag','K_in','K_hh','T_ag','T_in','T_hh','R','S_K','S_T','sK','sT','sKi','sTi'};
offset = struct();
for i = 1:numel(var_names)
    offset.(var_names{i}) = (i-1)*n_months;
end
n_vars = numel(var_names)*n_months;

% Objective
f = create_objective(n_vars, offset, economic_params, region_weights, n_months);

% Constraints
[Aineq, bineq, Aeq, beq] = build_constraints(reservoirs, offset, n_months, sla);

% Bounds
[lb, ub] = set_bounds(reservoirs, offset, n_months, n_vars, env_flow, sla, hh_min_K, hh_min_T);

% Solve
opts = optimoptions('linprog','Display','none');
[x, fval] = linprog(f, Aineq, bineq, Aeq, beq, lb, ub, opts);

% Results
res = process_results(x, offset, n_months);
fprintf('Total annual net benefit: ₹%.2f crore\n', fval/1e7);

%% plots (same as before) ...
% [Your bar, line, area, heatmap, sankey, etc.]
%%=== Visualization ===%%

% 1) Monthly Allocations & Transfer (grouped bar)
figure;
bar(months, [res.K_ag, res.K_hh, res.K_in, res.T_ag, res.T_hh, res.T_in, res.R], 'grouped');
legend('K Agri','K House','K Ind','T Agri','T House','T Ind','Transfer','Location','NorthWest');
xticks(months); xticklabels(months_lbl);
title('Monthly Allocations & Transfer'); ylabel('BCM'); grid on;

% 2) Storage vs Capacity: Karnataka
figure;
plot(months, res.S_K, 'b-o','LineWidth',1.5); hold on;
yline(reservoirs.K.Capacity, 'b--', 'K Capacity','LineWidth',1);
xticks(months); xticklabels(months_lbl);
xlabel('Month'); ylabel('Storage (BCM)');
title('Karnataka Reservoir Storage vs Capacity'); grid on;

% 3) Storage vs Capacity: Tamil Nadu
figure;
plot(months, res.S_T, 'r-s','LineWidth',1.5); hold on;
yline(reservoirs.T.Capacity, 'r--', 'T Capacity','LineWidth',1);
xticks(months); xticklabels(months_lbl);
xlabel('Month'); ylabel('Storage (BCM)');
title('Tamil Nadu Reservoir Storage vs Capacity'); grid on;

% 4) Stacked Area: Monthly Water Allocations & Transfer
figure;
area(1:12, [res.K_ag, res.K_in, res.K_hh, res.R, res.T_ag, res.T_in, res.T_hh]);
legend('K Agri','K Ind','K House','Transfer','T Agri','T Ind','T House','Location','EastOutside');
xticks(1:12); xticklabels(months_lbl);
title('Stacked Area: Monthly Water Allocations & Transfer'); ylabel('BCM'); grid on;

% 5) Monthly Shortages (stacked bar)
figure;
b = [res.sK, res.sKi, res.sT, res.sTi];
bar(1:12, b, 'stacked');
legend('Slack Ag K','Slack Ind K','Slack Ag T','Slack Ind T','Location','NorthEast');
xticks(1:12); xticklabels(months_lbl);
title('Monthly Agricultural & Industrial Shortages'); ylabel('BCM'); grid on;

% 6) Heatmap of Service Levels (Allocation/Demand)
SL = zeros(12,4);
SL(:,1) = res.K_ag ./ reservoirs.K.DemandAgri';
SL(:,2) = res.K_in ./ reservoirs.K.DemandInd';
SL(:,3) = res.T_ag ./ reservoirs.T.DemandAgri';
SL(:,4) = res.T_in ./ reservoirs.T.DemandInd';
sector_lbl = {'K Agri','K Ind','T Agri','T Ind'};
figure;
heatmap(months_lbl, sector_lbl, SL', 'ColorbarVisible','on');
title('Heatmap: Service Levels (Allocation/Demand)'); xlabel('Month'); ylabel('Sector');


%%--- Line Diagrams: Sector Profits & Total Benefit ---%%

% Agricultural Profit
agri_vals_K = economic_params.agri_K(:);
agri_vals_T = economic_params.agri_T(:);
agri_profit_K = region_weights(1) * agri_vals_K .* res.K_ag;
agri_profit_T = region_weights(2) * agri_vals_T .* res.T_ag;

figure;
plot(months, agri_profit_K, '-go','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','g'); hold on;
plot(months, agri_profit_T, '-mo','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','m');
xticks(months); xticklabels(months_lbl);
xlabel('Month'); ylabel('Profit (₹)');
title('Monthly Agricultural Profit: Karnataka vs Tamil Nadu');
legend('Karnataka','Tamil Nadu','Location','NorthWest'); grid on;

% Industrial Profit
ind_profit_K = region_weights(1)*economic_params.ind(1) * res.K_in;
ind_profit_T = region_weights(2)*economic_params.ind(2) * res.T_in;
figure;
plot(months, ind_profit_K, '-co','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','c'); hold on;
plot(months, ind_profit_T, '-bo','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','b');
xticks(months); xticklabels(months_lbl);
xlabel('Month'); ylabel('Profit (₹)');
title('Monthly Industrial Profit: Karnataka vs Tamil Nadu');
legend('Karnataka','Tamil Nadu','Location','NorthWest'); grid on;

% Total Monthly Benefit
total_benefit = agri_profit_K + agri_profit_T + ind_profit_K + ind_profit_T 
+ economic_params.transfer*res.R + region_weights(1)*economic_params.storage(1)*res.S_K 
+ region_weights(2)*economic_params.storage(2)*res.S_T - economic_params.penalty_agri*(res.sK+res.sT) 
- economic_params.penalty_ind*(res.sKi+res.sTi);
figure;
plot(months, total_benefit, '-mo','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','m');
xticks(months); xticklabels(months_lbl);
xlabel('Month'); ylabel('Benefit (₹)');
title('Total Monthly Benefit'); grid on;


%%--- Functions ---%%
function res = init_reservoir_params(name, cap, s0)
    res.Name        = name;
    res.Capacity    = cap;
    res.InitStorage = s0;
    switch name
      case 'Karnataka'
        res.Inflow     = [0.052161,0.043307,0.025151,0.031029,0.031029,0.657338,0.031029,2.817014,0.031029,1.901920,0.171738,0.161172];
        res.Evap       = [0.4768,0.2477,0.1306,0.3482,1.6144,2.1455,1.9850,2.8208,2.5196,2.8117,1.9204,1.0842];
        res.DemandAgri = [0.84942,1.41615,1.84035,1.84035,1.98252,1.84035,1.55781,1.27458,1.27458,0.99189,0.84942,0.84942];
        res.DemandInd  = repmat(0.0473,1,12);
      case 'Tamil Nadu'
        res.Inflow     = [0.064365,0.014808,0.012054,0.067341,0.067341,0.268025,0.067341,1.749602,0.067341,1.400768,0.411934,0.452935];
        res.Evap       = [0.1997,0.1170,0.0606,0.0809,0.7616,0.9879,0.7723,1.4767,0.9255,1.3932,0.9879,0.8317];
        res.DemandAgri = [0.93309,1.49742,2.06181,2.34654,2.06181,1.77903,1.21536,0.93309,0.93309,0.93309,0.93309,0.03078];
        res.DemandInd  = repmat(0.0933,1,12);
    end
end

function f = create_objective(nv, off, econ, w, nm)
    f = zeros(nv,1);
    for t=1:nm
        f(off.K_ag + t) = -w(1)*econ.agri_K(t);
        f(off.T_ag + t) = -w(2)*econ.agri_T(t);
    end
    f(off.K_in + (1:nm)) = -w(1)*econ.ind(1);
    f(off.T_in + (1:nm)) = -w(2)*econ.ind(2);
    f(off.K_hh + (1:nm)) = 0;
    f(off.T_hh + (1:nm)) = 0;
    f(off.R    + (1:nm)) = -econ.transfer;
    f(off.S_K  + (1:nm)) = -w(1)*econ.storage(1);
    f(off.S_T  + (1:nm)) = -w(2)*econ.storage(2);
    f(off.sK   + (1:nm)) =  econ.penalty_agri;
    f(off.sT   + (1:nm)) =  econ.penalty_agri;
    f(off.sKi  + (1:nm)) =  econ.penalty_ind;
    f(off.sTi  + (1:nm)) =  econ.penalty_ind;
end

function [Ai, bi, Ae, be] = build_constraints(res, off, nm, sla)
    nv = numel(fieldnames(off)) * nm;
    Ai=[]; bi=[]; Ae=[]; be=[];
    for m=1:nm
      %% Karnataka mass‐balance
      eq = zeros(1,nv);
      idxS = off.S_K + m;       
      if m>1, eq(off.S_K + m - 1) = -1; end
      eq(idxS) = 1;
      eq([off.K_ag+m, off.K_hh+m, off.K_in+m, off.R+m]) = -1;
      Ae = [Ae; eq];
      be = [be; res.K.InitStorage*(m==1) + res.K.Inflow(m) - res.K.Evap(m)];

      %% Tamil Nadu mass‐balance
      eq = zeros(1,nv);
      idxS = off.S_T + m;
      if m>1, eq(off.S_T + m -1) = -1; end
      eq(idxS) = 1;
      eq([off.T_ag+m, off.T_hh+m, off.T_in+m]) = -1;
      eq(off.R+m) = 1;
      Ae = [Ae; eq];
      be = [be; res.T.InitStorage*(m==1) + res.T.Inflow(m) - res.T.Evap(m) - 0.28317];

      %% Shortage = Demand for each sector (explicit)
      % K Agri
      eq = zeros(1,nv);
      eq(off.K_ag+m) = 1;  eq(off.sK+m) = 1;
      Ae = [Ae; eq];  be = [be; res.K.DemandAgri(m)];
      % T Agri
      eq = zeros(1,nv);
      eq(off.T_ag+m) = 1;  eq(off.sT+m) = 1;
      Ae = [Ae; eq];  be = [be; res.T.DemandAgri(m)];
      % K Ind
      eq = zeros(1,nv);
      eq(off.K_in+m) = 1; eq(off.sKi+m) = 1;
      Ae = [Ae; eq];  be = [be; res.K.DemandInd(m)];
      % T Ind
      eq = zeros(1,nv);
      eq(off.T_in+m) = 1; eq(off.sTi+m) = 1;
      Ae = [Ae; eq];  be = [be; res.T.DemandInd(m)];

      %% Service‐level inequalities
      % K Ind
      ineq = zeros(1,nv);
      ineq(off.K_in+m) = -1; ineq(off.sKi+m) = -1;
      Ai = [Ai; ineq]; bi = [bi; -sla.ind_min*res.K.DemandInd(m)];
      % T Ind
      ineq = zeros(1,nv);
      ineq(off.T_in+m) = -1; ineq(off.sTi+m) = -1;
      Ai = [Ai; ineq]; bi = [bi; -sla.ind_min*res.T.DemandInd(m)];
      % K Agri
      ineq = zeros(1,nv);
      ineq(off.K_ag+m) = -1; ineq(off.sK+m) = -1;
      Ai = [Ai; ineq]; bi = [bi; -sla.agri_min*res.K.DemandAgri(m)];
      % T Agri
      ineq = zeros(1,nv);
      ineq(off.T_ag+m) = -1; ineq(off.sT+m) = -1;
      Ai = [Ai; ineq]; bi = [bi; -sla.agri_min*res.T.DemandAgri(m)];
    end
end

function [lb, ub] = set_bounds(res, off, nm, nv, env_flow, sla, hh_min_K, hh_min_T)
    lb = zeros(nv,1);
    ub = inf(nv,1);
    for m = 1:nm
        ub(off.S_K+m)    = res.K.Capacity;
        ub(off.S_T+m)    = res.T.Capacity;
        ub(off.K_ag+m)   = res.K.DemandAgri(m);
        ub(off.T_ag+m)   = res.T.DemandAgri(m);
        ub(off.K_in+m)   = res.K.DemandInd(m);
        ub(off.T_in+m)   = res.T.DemandInd(m);
        lb(off.R+m)      = env_flow;
        lb(off.K_hh+m)   = hh_min_K(m);
        lb(off.T_hh+m)   = hh_min_T(m);
    end
end

function res = process_results(x, off, nm)
    idx = 1:nm;
    res.K_ag  = x(off.K_ag + idx);
    res.K_hh  = x(off.K_hh + idx);
    res.K_in  = x(off.K_in + idx);
    res.T_ag  = x(off.T_ag + idx);
    res.T_hh  = x(off.T_hh + idx);
    res.T_in  = x(off.T_in + idx);
    res.R     = x(off.R    + idx);
    res.S_K   = x(off.S_K  + idx);
    res.S_T   = x(off.S_T  + idx);
    res.sK    = x(off.sK   + idx);
    res.sT    = x(off.sT   + idx);
    res.sKi   = x(off.sKi  + idx);
    res.sTi   = x(off.sTi  + idx);
end
