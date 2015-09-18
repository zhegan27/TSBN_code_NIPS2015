
visible = vgen';
postprocess;

%Plot a representation of the weights
% hdl = figure(4); weightreport
% set(hdl,'Name','Layer 2 CRBM weights');

%Plot top-layer activations
% figure(5); imagesc(hidden2'); colormap gray;
% title('Top hidden layer, activations'); ylabel('hidden units'); xlabel('frames')
% %Plot middle-layer probabilities
% figure(6); imagesc(hidden1'); colormap gray;
% title('First hidden layer, probabilities'); ylabel('hidden units'); xlabel('frames')

fprintf(1,'Playing generated sequence\n');
 figure(2); expPlayData(skel, newdata, 1/30)
% figure, expPlotData( skel, newdata, 1/30 )
% 
% hFig = figure(1);
% fps = 10;
% set(hFig, 'Position', [200 200 400 400]) %[left bottom width height]
% filename = 'mocap_walking.gif'; 
% for n = 1:100
%     expPlayData(skel, newdata, 1/30);
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     if n == 1;
%       imwrite(imind,cm,filename,'gif','Loopcount',inf);
%     else
%       imwrite(imind,cm,filename,'gif','WriteMode','append');
%     end
% end

