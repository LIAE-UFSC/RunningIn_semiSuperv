figure;

for k = 1:length(N)
    apr(k,1) = nnz(metricsOk.N==N(k))/nnz(metrics.N==N(k));
    apr(k,2) = nnz(metricsNotOk.N==N(k))/nnz(metrics.N==N(k));
end
bar(apr,'stacked');
xticklabels(num2str(N));
ylabel("Accepted proportion")
xlabel("N")
title("Geral");
clear apr

for kM = 1:length(models)
    figure;
    for k = 1:length(Window)
        indAll = (metrics.N-1).*(metrics.D) == Window(k);
        indOk = (metricsOk.N-1).*(metricsOk.D) == Window(k);
        indNotOk = (metricsNotOk.N-1).*(metricsNotOk.D) == Window(k);
        apr(k,1) = nnz(indOk & strcmp(models{kM},metricsOk.Model))/nnz(indAll & strcmp(models{kM},metrics.Model));
        apr(k,2) = nnz(indNotOk & strcmp(models{kM},metricsNotOk.Model))/nnz(indAll & strcmp(models{kM},metrics.Model));
    end
    bar(Window, apr,'stacked');
    xlabel("Window")
    title(models{kM});
    clear apr
end