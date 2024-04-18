clear; close all;

path = "resultadosA";

N =  4;
D =  10;
M =  1;
MAvg = 10;

dataAll = importAllData(path);

dataAll = dataAll(dataAll.N==N & dataAll.D==D & dataAll.M==M,:);

metrics = checkValidity(dataAll);
metricsOk = metrics(metrics.MCC>=0.7 & metrics.AllUnitsOk,:);
metricsNotOk =  metrics(metrics.MCC<=0.7 | ~metrics.AllUnitsOk,:);

models = unique(metrics.Model);

Mlim = [-Inf, Inf];
Nlim = [-Inf, Inf];
Dlim = [-Inf, Inf];

%% % aprovado


modelNames = {
    {'KNN3'
    'KNN6'
    'KNN30'
    'KNN60'
    'KNN300'}
    {'RandomForest10'
    'RandomForest100'
    'RandomForest1000'}
    {'SVM_Linear'
    'SVM_Poly2'
    'SVM_Poly3'
    'SVM_rbf'}
    };

legendsModel = {
    {'K = 3'
    'K = 6'
    'K = 30'
    'K = 60'
    'K = 300'}
    {'n_T = 10'
    'n_T = 100'
    'n_T = 1000'}
    {'Linear kernel'
    'Quadratic kernel'
    'Cubic kernel'
    'RBF kernel'}
    };

fName = {'KNN', 'RF','SVM'};

for k1 = 1:length(modelNames)
    fig = figure;
    fig.Position = [222 243 1388 153];

    for kM = 1:length(modelNames{k1})
        tableData = mixResults(dataAll,modelNames{k1}{kM},MAvg,Mlim,Nlim,Dlim);
        data = tableData{:,end};
        time = tableData.Time;
        un_val = unique(tableData.Unidade);
        for k2 = 1:length(un_val)
            unidade = un_val(k2);
            ind = tableData.Unidade == unidade & tableData.N_ensaio == 0;
            subplot(1,length(un_val),k2)
            hold on
            plot(time(ind),data(ind))
            hold off
            title(strcat("Unidade ",un_val{k2}))
            ylim([0,1])
        end
    end
    legend(legendsModel{k1},"Location","best");
    set(gcf, 'Color', 'w');
    tightfig();
    export_fig(strcat("ResultadosAnaliseParametros\ResultadoM",num2str(M),"N",num2str(N),"D",num2str(D),"model",fName{k1}),"-pdf")
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