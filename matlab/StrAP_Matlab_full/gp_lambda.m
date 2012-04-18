function [current_lambda, xstar, mu,S2,idx]= gp_lambda(pts_action,range)

covfunc = {'covSum', {'covSEiso','covNoise'}};
loghyper = [log(1.0); log(1.0); log(0.1)];
% loghyper = [log(0.3); log(1.08); log(5e-5)];


%xstar = linspace(40,120,500)';

xstar = rand(2000, 1) .* (range(2)-range(1)) + range(1);
[mu S2] = gpr(loghyper, covfunc, pts_action(:,1), pts_action(:,2), xstar);

[mx, idx] = max(mu + S2);

current_lambda =  xstar(idx);

end
