% set the activation function you want before, e.g.
% params.actFunc = 'sigmoid';

switch params.actFunc
    case 'linthresh',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f     = @linthresh;
        params.df    = @linthreshg;
    case 'linthresh1',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f     = @linthresh1;
        params.df    = @linthreshg1;
    case 'identity',
        params.f     = @identity;
        params.df    = @identityg;
    case 'threshold',
        threshold = 11;
        params.f     = @(x) (x>threshold);
        params.df    = @(x) (x>threshold);
    case 'tanh',
        params.f = @(x) (tanh(x));
        params.df = @(z) (1-z.^2);
    case 'sclTanh',
        % see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        params.f = @(x) (1.7159*tanh(2/3 * x));
        params.df = @(z) (1.7159 * 2/3 * (1-(2/3 * z).^2));
%     case 'sclTanhTwist',
%         % see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
%         params.f = @(x) (1.7159*tanh(2/3 * x) + 0.01*x);
%         params.df = @(z) (1.7159 * 2/3 * (1-(2/3 * z).^2) +0.01);
    case 'sigmoid',
        params.f = @(x) (1./(1 + exp(-x)));
        params.df = @(z) (z .* (1 - z));
    case 'rectMax',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f = @(x) (max(0,x));
        params.df = @(z) (z>0);
        params.normalizeWe = 0;
    case 'rectLog1Exp',
        params.f = @(x) (log(1+exp(x)));
        params.df = @(z) (1./(1 + exp(-z)));
        params.normalizeWe = 0;
    otherwise
        error('Define an activation function, dude!')
end