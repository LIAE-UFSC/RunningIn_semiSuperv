clear;

close all;

path = "resultadosA";

dataAllA = importAllData(path);

path = "resultadosB";

dataAllB = importAllData(path);

dataAllA = sortrows(dataAllA,{'N','M','D'});
dataAllB = sortrows(dataAllB,{'N','M','D'});


for k = 1:height(dataAllA)
    
    if dataAllA{k,"N"} ~= dataAllB{k,"N"}
        error("Unequal parameters")
    end
    if dataAllA{k,"M"} ~= dataAllB{k,"M"}
        error("Unequal parameters")
    end
    if dataAllA{k,"D"} ~= dataAllB{k,"D"}
        error("Unequal parameters")
    end

    dataAllA{k,"Tempo"}{1} = [dataAllA{k,"Tempo"}{1};dataAllB{k,"Tempo"}{1}];
    dataAllA{k,"N_ensaio"}{1} = [dataAllA{k,"N_ensaio"}{1};dataAllB{k,"N_ensaio"}{1}];
    dataAllA{k,"Unidade"}{1} = [dataAllA{k,"Unidade"}{1};dataAllB{k,"Unidade"}{1}];
    dataAllA{k,"RunIn"}{1} = [dataAllA{k,"RunIn"}{1};dataAllB{k,"RunIn"}{1}];

    models = fieldnames(dataAllA{k,"Results"});
    for kM = 1:length(models)
        dataAllA{k,"Results"}.(models{kM}) = [dataAllA{k,"Results"}.(models{kM});dataAllB{k,"Results"}.(models{kM})];
    end
end

ind = (dataAllA.N>1) | ((dataAllA.N==1) & (dataAllA.D==1));

dataAll = dataAllA(ind,:);

clear dataAllA dataAllB

metrics = checkValidity(dataAll);
metricsOk = metrics(metrics.MCC>=0.7 & metrics.AllUnitsOk,:);
metricsNotOk =  metrics(metrics.MCC<=0.7 | ~metrics.AllUnitsOk,:);

models = unique(metrics.Model);
N =  unique(metrics.N);
D =  unique(metrics.D);
M =  unique(metrics.M);

MAvg = 60;
Mlim = [-Inf, Inf];
Nlim = [-Inf, Inf];
Dlim = [-Inf, Inf];

%% % aprovado

fig = figure;
fig.Position = [377 183 1949 795];

subplot(3,length(models)+1,1)

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
    subplot(3,length(models)+1,kM+1)
    for k = 1:length(N)
        apr(k,1) = nnz(metricsOk.N==N(k) & strcmp(models{kM},metricsOk.Model))/nnz(metrics.N==N(k) & strcmp(models{kM},metrics.Model));
        apr(k,2) = nnz(metricsNotOk.N==N(k) & strcmp(models{kM},metricsNotOk.Model))/nnz(metrics.N==N(k) & strcmp(models{kM},metrics.Model));
    end
    bar(apr,'stacked');
    xticklabels(num2str(N));
    xlabel("N")
    title(models{kM});
    clear apr
end

subplot(3,length(models)+1,length(models)+2)

for k = 1:length(D)
    apr(k,1) = nnz(metricsOk.D==D(k))/nnz(metrics.D==D(k));
    apr(k,2) = nnz(metricsNotOk.D==D(k))/nnz(metrics.D==D(k));
end
bar(D,apr,'stacked');
ylabel("Accepted proportion")
xlabel("D")
title("Geral");
clear apr

for kM = 1:length(models)
    subplot(3,length(models)+1,length(models)+kM+2)
    for k = 1:length(D)
        apr(k,1) = nnz(metricsOk.D==D(k) & strcmp(models{kM},metricsOk.Model))/nnz(metrics.D==D(k) & strcmp(models{kM},metrics.Model));
        apr(k,2) = nnz(metricsNotOk.D==D(k) & strcmp(models{kM},metricsNotOk.Model))/nnz(metrics.D==D(k) & strcmp(models{kM},metrics.Model));
    end
    bar(D,apr,'stacked');
    xlabel("D")
    title(models{kM});
    clear apr
end

subplot(3,length(models)+1,2*length(models)+3)

Window = unique((N-1)*D');

for k = 1:length(Window)
    indAll = (metrics.N-1).*(metrics.D) == Window(k);
    indOk = (metricsOk.N-1).*(metricsOk.D) == Window(k);
    indNotOk = (metricsNotOk.N-1).*(metricsNotOk.D) == Window(k);
    if any(indAll)
        apr(k,1) = nnz(indOk)/nnz(indAll);
        apr(k,2) = nnz(indNotOk)/nnz(indAll);
    end
end
Window = Window(1:size(apr,1));
bar(apr,'stacked');
ylabel("Accepted proportion")
xlabel("Window")
title("Geral");
clear apr

for kM = 1:length(models)
    subplot(3,length(models)+1,2*length(models)+kM+3)
    for k = 1:length(Window)
        indAll = (metrics.N-1).*(metrics.D) == Window(k);
        indOk = (metricsOk.N-1).*(metricsOk.D) == Window(k);
        indNotOk = (metricsNotOk.N-1).*(metricsNotOk.D) == Window(k);
        apr(k,1) = nnz(indOk & strcmp(models{kM},metricsOk.Model))/nnz(indAll & strcmp(models{kM},metrics.Model));
        apr(k,2) = nnz(indNotOk & strcmp(models{kM},metricsNotOk.Model))/nnz(indAll & strcmp(models{kM},metrics.Model));
    end
    bar(apr,'stacked');
    xlabel("Window")
    title(models{kM});
    clear apr
end

set(gcf, 'Color', 'w');
tightfig();
% export_fig("ResultadosAnaliseParametros\analise%aprov","-pdf")
% close;

%% Variando N

modelNames = unique(metricsOk.Model);

for kM = 1:length(modelNames)
    tableData = mixResults(dataAll,modelNames{kM},MAvg,Mlim,Nlim,Dlim);

    un_val = unique(tableData.Unidade);

    f = figure;
    f.Position	= [46 20 2416 958];
    for k2 = 1:length(un_val)
        [dataOut,M,N,D,time] = separaND_unidade(tableData,un_val(k2),0);
        for k3 = 1:length(N)
            subplot(length(un_val),length(N),(k2-1)*length(N)+k3)
            Z = squeeze(dataOut(1,k3,:,:));
            q = quantile(Z,3);
            q(1,:) = q(1,:)-q(2,:);
            q(3,:) = q(2,:)-q(3,:);
            er = q([3,1],:);
            boundedline(time',q(2,:)',er','alpha');
            title(strcat("Unidade ",un_val{k2}," (N = ", num2str(N(k3)), ")"))
        end
    end

    set(gcf, 'Color', 'w');
    tightfig();
    export_fig(strcat("ResultadosAnaliseParametros\variaN",modelNames{kM}),"-pdf")
    close;
end

%% Variando D

modelNames = unique(metricsOk.Model);

for kM = 1:length(modelNames)
    tableData = mixResults(dataAll,modelNames{kM},MAvg,Mlim,Nlim,Dlim);

    un_val = unique(tableData.Unidade);

    f = figure;
    f.Position	= [46 20 2416 958];
    for k2 = 1:length(un_val)
        [dataOut,M,N,D,time] = separaND_unidade(tableData,un_val(k2),0);
        for k3 = 1:length(D)
            subplot(length(un_val),length(D),(k2-1)*length(D)+k3)
            Z = squeeze(dataOut(1,:,k3,:));
            q = quantile(Z,3);
            q(1,:) = q(1,:)-q(2,:);
            q(3,:) = q(2,:)-q(3,:);
            er = q([3,1],:);
            boundedline(time',q(2,:)',er','alpha');
            title(strcat("Unidade ",un_val{k2}," (D = ", num2str(D(k3)), ")"))
        end
    end

    set(gcf, 'Color', 'w');
    tightfig();
    export_fig(strcat("ResultadosAnaliseParametros\variaD",modelNames{kM}),"-pdf")
    close;
end

%% Func

function dataAll = importAllData(path)

folders = dir(path);
folders = folders([folders.isdir]);

modelNames = {folders(3:end).name};

folders = fullfile(path,modelNames);
dataAll = cell2table(cell(0,8), 'VariableNames', {'N','D','M','Tempo','N_ensaio','Unidade','RunIn','Results'});

for kM = 1:length(modelNames)
    fileList = dir(fullfile(folders{kM},'*.csv'));
    for kF = 1:length(fileList)
        N = str2double(extractBetween(fileList(kF).name,"N","D"));
        D = str2double(extractBetween(fileList(kF).name,"D","M"));
        M = str2double(extractBetween(fileList(kF).name,"M","CLASS"));

        if length(N)>1
            N = N(1);
        end

        if length(D)>1
            D = D(1);
        end

        if length(M)>1
            M = M(1);
        end

        data = importfile(fullfile(folders{kM},fileList(kF).name));

        if ~isempty(dataAll) && any(dataAll.N == N & dataAll.D == D & dataAll.M == M)
            resultsTemp = dataAll{dataAll.N == N & dataAll.D == D & dataAll.M == M,"Results"}{1};
            resultsTemp.(modelNames{kM}) = data.runinEst;
            dataAll{dataAll.N == N & dataAll.D == D & dataAll.M == M,"Results"} = {resultsTemp};
        else
            resultsTemp = struct;
            resultsTemp.(modelNames{kM}) = data.runinEst;
            row = struct;
            row.N = N;
            row.D = D;
            row.M = M;
            row.Tempo = data.Tempo;
            row.N_ensaio = data.N_ensaio;
            row.Unidade = data.Unidade;
            row.RunIn = data.runinOg;
            row.Results = {resultsTemp};
            dataAll = [dataAll;struct2table(row,"AsArray",true)];
        end
    end
end

for k = 1:height(dataAll)
    results(k) = dataAll{k,"Results"}{1};
end

dataAll.Results = results';

end

function data = importfile(filename)

dataLines = [2, Inf];

opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "Tempo", "N_ensaio", "Unidade", "runinEst", "runinOg"];
opts.SelectedVariableNames = ["Tempo", "N_ensaio", "Unidade", "runinEst", "runinOg"];
opts.VariableTypes = ["string", "double", "double", "string", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
% opts = setvaropts(opts, "Unidade", "TrimNonNumeric", true);
% opts = setvaropts(opts, "Unidade", "ThousandsSeparator", ",");

% Import the data
data = readtable(filename, opts);

end