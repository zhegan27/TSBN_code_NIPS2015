clear; clc; close all;
addpath(genpath('.'));

%% dataset & results
load bouncing_balls_testing_data;
N = length(Data);
[M,T]=size(Data{1}');

v = Data{21}';

hFig = figure(1);
set(hFig, 'Position', [400 400 400 400]) %[left bottom width height]
filename = 'bouncing_ball.gif'; 
for n = 1:100
    imagesc(reshape(v(:,n),30,30));colormap gray; title('training sample');
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if n == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end



