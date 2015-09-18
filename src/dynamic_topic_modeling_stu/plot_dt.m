% plot topics

load('dtsbn_topics.mat');

[K, T] = size(h);
id_topics = [29, 30, 130];
yy = h(id_topics,:);
styles = {'k-','r-','b-','g-'};



for i = 1:3
    figure(i);
    usage = yy(i,:);
    usage = (usage - min(usage)) / (max(usage) - min(usage));
    xx = 1790: 1790+T-1; 
    plot(xx,usage,styles{i},'LineWidth',3,'MarkerSize',3);
    set(gca,'XTick',[1800 ,1850, 1900, 1950, 2000]); 
    labels = ['1800'; '1850'; '1900'; '1950';'2000'];
    set(gca,'XTickLabel',labels, 'FontSize', 15)
    % set(gca,'xtick',xx(1:80:224),'xticklabel',num2str(xx(1:80:224)) ,'FontSize', 20);
    legend(['Topic ', num2str(id_topics(i))],'FontSize', 20, 'Location', 'North'); grid on;
    axis([1790, 1790+T-1,-0.05,1.05 ]);
end