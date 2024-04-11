function [MCC,M,N,D,F] = calculaMCC(data)

M = unique(data.M);
N = unique(data.N);
D = unique(data.D);

models = fieldnames(data{1,"Results"});

for kModel = 1:length(models)
    mccTemp.(models{kModel}) = NaN;
end

MCC = repmat(mccTemp,length(M),length(N),length(D));
F = repmat(mccTemp,length(M),length(N),length(D));

for kM = 1:length(M)
    for kN = 1:length(N)
        for kD = 1:length(D)
            ind = data.M==M(kM) & data.N==N(kN) & data.D==D(kD);

            if ~any(ind)
                break
            end

            results = data{ind,"Results"};
            real = data{ind,"RunIn"}{1};
            
            for kModel = 1:length(models)
                resultModel = results.(models{kModel});
                mat = confusionmat(real(real~=-1),resultModel(real~=-1));
                TP = mat(1,1);
                TN = mat(2,2);
                FP = mat(2,1);
                FN = mat(1,2);
                MCC(kM,kN,kD).(models{kModel}) = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
                F(kM,kN,kD).(models{kModel}) = FP+FN;
            end
        end
    end
end

end