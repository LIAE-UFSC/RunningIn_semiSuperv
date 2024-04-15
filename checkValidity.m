function [conditions] = checkValidity(dataAll)

allColumns = {'N','M','D','Model','MCC','AllUnitsOk'};

for k = 1:height(dataAll)

    real = dataAll{k,"RunIn"}{1};
    models = fieldnames(dataAll{k,"Results"});
    n_ensaio = dataAll{k,"N_ensaio"}{1};
    unidade = dataAll{k,"Unidade"}{1};
    for kM = 1:length(models)
        temp(kM).N = dataAll{k,"N"};
        temp(kM).D = dataAll{k,"D"};
        temp(kM).M = dataAll{k,"M"};
        temp(kM).Model = models{kM};
        resultModel = dataAll{k,"Results"}.(models{kM});
        mat = confusionmat(real(real~=-1),resultModel(real~=-1));
        TP = mat(1,1);
        TN = mat(2,2);
        FP = mat(2,1);
        FN = mat(1,2);
        temp(kM).MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
        temp(kM).F = FP+FN;
        
        indAm = n_ensaio==0 & real==-1;
        un = unique(unidade(indAm));
        for kU = 1:length(un)
            if kU ==1
                temp(kM).AllUnitsOk=true;
                temp(kM).NotOkay={};
            end
            if all(resultModel(strcmp(un{kU},unidade) & indAm) == 0) || ...
                    all(resultModel(strcmp(un{kU},unidade) & indAm) == 1)
                temp(kM).AllUnitsOk = false;
                temp(kM).NotOkay = [temp(kM).NotOkay, un{kU}];
            end
            
        end
    end

    if k==1
        conditions = struct2table(temp);
    else
        conditions = [conditions;struct2table(temp)];
    end
end

end