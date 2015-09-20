function expPlotData(skel, channels, frameLength, xlim, ylim, zlim ) %#ok

expVisualise( channels(1, :), skel );
axis equal

% limits
xlim = get(gca, 'xlim');
minY1 = xlim(1);
maxY1 = xlim(2);
ylim = get(gca, 'ylim');
minY3 = ylim(1);
maxY3 = ylim(2);
zlim = get(gca, 'zlim');
minY2 = zlim(1);
maxY2 = zlim(2);
for ii = 1:size(channels, 1)
	Y = exp2xyz(skel, channels(ii, :));
	minY1 = min([Y(:, 1); minY1]);
	minY2 = min([Y(:, 2); minY2]);
	minY3 = min([Y(:, 3); minY3]);
	maxY1 = max([Y(:, 1); maxY1]);
	maxY2 = max([Y(:, 2); maxY2]);
	maxY3 = max([Y(:, 3); maxY3]);
end
xlim = [minY1 maxY1];
ylim = [minY3 maxY3];
zlim = [minY2 maxY2];
set( gca, 'xlim', xlim, 'ylim', ylim, 'zlim', zlim );

% axis
ha = gca;
ha.YTick = []; ha.XTick = []; ha.ZTick = []; ha.Box = 'on';
ha.XLabel = []; ha.YLabel = []; ha.ZLabel = [];

% trace
val = zeros( 18, 3, 400 );
for i=1:400
	val(:,:,i) = exp2xyz( skel, channels(i,:) );
end

hold on
which = [ 1 8 15 ];
plot3( squeeze( val(which,1,:) )', squeeze( val(which,3,:) )', squeeze( val(which,2,:) )', '-', 'linewidth', 1 )

for jj = [ 50 100 150 200 250 300 350 400 ]
	expVisualise( channels(jj, :), skel );
end
hold off

end

function handle = expVisualise(channels, skel, padding)


if nargin<3
    padding = 0;
end

channels = [channels zeros(1, padding)];
vals = exp2xyz(skel, channels);
connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);

handle(1) = plot3( vals(:, 1), vals(:, 3), vals(:, 2), '.' );
set( handle(1), 'markersize', 20, 'color', [ 0.4 0.4 0.4 ] );
% hold on
% grid on
for i = 1:length(indices)
    handle(i+1) = line([vals(I(i), 1) vals(J(i), 1)], ...
        [vals(I(i), 3) vals(J(i), 3)], ...
        [vals(I(i), 2) vals(J(i), 2)]);
    set( handle(i+1), 'linewidth', 2, 'color', [ 0.1 0.1 0.1 ] );
end
% axis equal
% xlabel('x')
% ylabel('z')
% zlabel('y')
% axis on

end

