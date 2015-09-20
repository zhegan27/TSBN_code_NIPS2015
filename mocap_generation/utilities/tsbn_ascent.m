function [parameters,result]= tsbn_ascent(v,parameters,opts)

% Output Results
logpv=[]; predErr = []; recErr = [];

P=numel(parameters); prevMean = 0; prevVar = 0;
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
    [grads,~,meanll,varll] = tsbn_gradient(vB,parameters,prevMean,prevVar);
    
    prevMean = meanll;
    prevVar = varll;
    
    if opts.method == 2
        % Update Parameters
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
        % Update Parameters
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
        % Update Parameters
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
        Ntest = length(v); tmp1 = 0; tmp2 = 0; tmp3 = 0;
        for i = 1:Ntest
            vB = v{i}';
            tmp1 = tmp1 + tsbn_calc_loglike(vB,parameters)/Ntest;
            [tt2,tt3] = tsbn_calc_pred_error(vB,parameters);
            tmp2 = tmp2 + tt2/Ntest;
            tmp3 = tmp3 + tt3/Ntest;
        end;

        logpv=[logpv;tmp1]; recErr = [recErr;tmp2];
        predErr = [predErr;tmp3];
        
        totaltime = toc;
        fprintf('Test, Iter %d: logpv=%4.8f, recErr=%4.8f, predErr=%4.8f, time=%4.8f\n',...
            iter,logpv(end),recErr(end),predErr(end),totaltime);
    end
end
result.logpv=logpv; 
result.predErr = predErr; 
result.recErr = recErr;

end