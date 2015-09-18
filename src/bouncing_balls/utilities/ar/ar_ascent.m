function [parameters,result]=ar_ascent(v,parameters,opts,vtest)

% Output Results
logpv_test=[]; predErr_test = [];

P=numel(parameters);
N = length(v);

rmsdecay=opts.rmsdecay;
momentum=parameters; 
for p=1:P
    momentum{p}(:)=0;
end

rms=cell(P); g=cell(P);
total_grads = cell(P);

tic
for iter=1:opts.iters
    % Get Gradient
    index = randperm(N);
    vB = v{index(1)}';
    grads = ar_gradient(vB,parameters);
    
    if opts.method == 2
        % Update Parameters using RMSprop method
        for p=1:P
            if iter == 1
                rms{p} = grads{p}.^2; g{p} = grads{p};
            else
                rms{p} = rmsdecay*rms{p} + (1-rmsdecay)*grads{p}.^2;
                g{p} = rmsdecay*g{p} + (1-rmsdecay)*grads{p};
            end;
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            step = step./(sqrt(rms{p}-g{p}.^2+1e-4));
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    elseif opts.method == 1
        % Update Parameters using AdaGrad method
        for p=1:P
            if iter == 1
                total_grads{p} = grads{p}.^2;
            else
                total_grads{p} = total_grads{p} + grads{p}.^2;
            end;
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            step = step./(sqrt(total_grads{p})+1e-6);
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    elseif opts.method == 0
        % Update Parameters using SGD method
        for p=1:P
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    end;

    % Check performance every so often
    if mod(iter,opts.evalInterval)==0
        Ntest = length(vtest); tmp1 = 0; tmp2 = 0; 
        for i = 1:Ntest
            vB = vtest{i}';
            tmp1 = tmp1 + ar_calc_loglike(vB,parameters)/Ntest;
            tmp2 = tmp2 + ar_calc_pred_error(vB,parameters)/Ntest;
        end;
        logpv_test=[logpv_test;tmp1]; 
        predErr_test = [predErr_test;tmp2]; totaltime = toc;
        fprintf('Test, Iter %d: logpv=%4.8f, predErr=%4.8f, time=%4.8f\n',...
            iter,logpv_test(end),predErr_test(end),totaltime);
    end
end
result.logpv_test=logpv_test; 
result.predErr_test = predErr_test; 
end