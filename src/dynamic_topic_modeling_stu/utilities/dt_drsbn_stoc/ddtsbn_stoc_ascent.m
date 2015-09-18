function [parameters,result]=ddtsbn_stoc_ascent(vtrain,parameters,opts,vhdout, vtest)

% Output Results
logpv_test=[]; predErr_test = []; recErr_test = [];
PR=[]; RC=[]; PP=[];

P=numel(parameters); prevMean = 0; prevVar = 0;
N = size(vtrain,2);

rmsdecay=opts.rmsdecay;
momentum=parameters; 
for p=1:P
    momentum{p}(:)=0;
end

rms=cell(P); g=cell(P);
total_grads = cell(P);

tic
for iter=1:opts.iters
    
%     if mod(iter,20)==0
%         fprintf('Test, Iter %d\n',iter);
%     end;
    
    % Get Gradient
    index = randperm(N);
    vB = vtrain; % v{index(1)}';
    [grads,~,meanll,varll] = dtsbn_stoc_gradient(vB,parameters,prevMean,prevVar);
    
    
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
%     if mod(iter,opts.evalInterval)==0
%         Ntest = length(vtest); %tmp1 = 0; tmp2 = 0; tmp3 = 0;
%         
%         index = randperm(Ntest);
%         vB = vtest{index(1)}';
%         tmp1 = drsbn_stoc_calc_loglike(vB,parameters);
%         [tmp2,tmp3] = drsbn_stoc_calc_pred_error(vB,parameters);
%         
%         logpv_test=[logpv_test;tmp1]; totaltime = toc;
%         recErr_test = [recErr_test;tmp2]; predErr_test = [predErr_test;tmp3]; 
%         fprintf('Test, Iter %d: logpv=%4.8f, recErr=%4.8f, predErr=%4.8f, time=%4.8f\n',...
%             iter,logpv_test(end),recErr_test(end),predErr_test(end),totaltime);
%     end
    
    % Check performance every so often
    if mod(iter,opts.evalInterval)==0      
        [lb, pr, rc] = dtsbn_stoc_calc_loglike(vhdout,vtrain,parameters);
        [rec_err, pp] = dtsbn_stoc_calc_pred_error(vtest,vtrain,parameters);
             
        logpv_test=[logpv_test;lb]; recErr_test = [recErr_test;rec_err];
        totaltime = toc;
        PR=[PR; pr]; RC=[RC; rc]; PP=[PP; pp];
        fprintf('Test, Iter %d: logpv=%4.4f, recErr=%4.4f, pr=%4.4f, rc=%4.4f, pp=%4.4f, time=%4.4f\n',...
            iter,logpv_test(end),recErr_test(end),...
            PR(end), RC(end), PP(end),...
            totaltime);
    end    
    
end
result.logpv_test=logpv_test;  
result.recErr_test = recErr_test;
result.PR = PR;
result.RC = RC;
result.PP = PP;
end