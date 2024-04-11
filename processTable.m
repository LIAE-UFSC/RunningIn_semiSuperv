clear

path = "resultados";

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

[MCC,M,N,D,F] = calculaMCC(dataAll);

MCC = squeeze(MCC);

sz = size(MCC);

FMax = 0;

for i = 1 :length(modelNames)
    FMax = max([FMax, [F.(modelNames{i})]]);
end


% for i = 1 :length(modelNames)
%     figure;
%     surf(N',D,reshape([F.(modelNames{i})],sz)');
%     view(2);
%     set(gca,"XScale","log")
%     title(modelNames{i})
%     colorbar; 
%     caxis([0,FMax]);
% end

MAvg = 60;
Mlim = [-Inf,Inf];
Nlim = [-Inf,Inf];
Dlim = [-Inf,Inf];


for kM = 1:length(modelNames)
     tableData = mixResults(dataAll,modelNames{kM},MAvg,Mlim,Nlim,Dlim);

     un_val = unique(tableData.Unidade);
%     
%     nens = tableData.N_ensaio;
%     amacEnsaios = tableData(nens==0,:);
%     amacEnsaios = removevars(amacEnsaios,{'N_ensaio'});
%     
%     un_val = unique(amacEnsaios.Unidade);
%     
%     figure;
%     
%     for k2 = 1:height(un_val)
%         subplot(2,2,k2)
%         dataUn = amacEnsaios(amacEnsaios.Unidade==un_val(k2),:);
%         dataUn = sortrows(dataUn,{'Time'});
%         time = dataUn.Time;
%         dataUn = removevars(dataUn,{'Time','Unidade','RunIn'});
%         class = table2array(dataUn);
%         q = quantile(class',4);
%         plot(time,q)
%         title(strcat("Unidade ",num2str(un_val(k2)), " (", modelNames{kM},")"))
%     end
    for k2 = 1:height(un_val)
        [dataOut,M,N,D,time] = separaND_unidade(tableData,un_val(k2),0);
    end
end

%% Func

function data = importfile(filename)

dataLines = [2, Inf];

opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "Tempo", "N_ensaio", "Unidade", "runinEst", "runinOg"];
opts.SelectedVariableNames = ["Tempo", "N_ensaio", "Unidade", "runinEst", "runinOg"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Unidade", "TrimNonNumeric", true);
opts = setvaropts(opts, "Unidade", "ThousandsSeparator", ",");

% Import the data
data = readtable(filename, opts);

end