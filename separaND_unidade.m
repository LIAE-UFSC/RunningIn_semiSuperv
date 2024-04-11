function [dataOut,M,N,D,time] = separaND_unidade(dataIn,unidade,ensaio)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

tab = sortrows(dataIn(dataIn.Unidade == unidade & dataIn.N_ensaio == ensaio,:),{'Time'});
time = tab.Time;
tab = removevars(tab,{'Time','Unidade','RunIn','N_ensaio'});

allStr = tab.Properties.VariableNames;

Nloc = regexp(allStr,"N");
Mloc = regexp(allStr,"M");
Dloc = regexp(allStr,"D");

for k = 1:length(allStr)
    varName = allStr{k};
    N(k) = str2double(varName(Nloc{k}+1:Dloc{k}-1));
    M(k) = str2double(varName(Mloc{k}+1:Nloc{k}-1));
    D(k) = str2double(varName(Dloc{k}+1:end));
end

N = sort(unique(N));
M = sort(unique(M));
D = sort(unique(D));

dataOut = nan(length(M),length(N),length(D),length(time));

for kM = 1:length(M)
    for kN = 1:length(N)
        for kD = 1:length(D)
            name = strcat("M",num2str(M(kM)),"N",num2str(N(kN)),"D",num2str(D(kD)));
            if any(strcmp(allStr,name))
                dataOut(kM,kN,kD,:) = tab{:,name};
            end

        end
    end
end

end