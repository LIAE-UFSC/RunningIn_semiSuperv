function [tableData] = mixResults(data,model,mvng_avg,Mlim,Nlim,Dlim)

if nargin <= 2
    mvng_avg = 1;
end

M = [];
N = [];
D = [];

ft = true;

for k = 1:height(data)
    if data{k,"M"}<Mlim(1) || data{k,"M"}>Mlim(2) || ...
       data{k,"N"}<Nlim(1) || data{k,"N"}>Nlim(2) || ...
       data{k,"D"}<Dlim(1) || data{k,"D"}>Dlim(2)
        continue
    end


    M = [M, data{k,"M"}];
    N = [N, data{k,"N"}];
    D = [D, data{k,"D"}];
    name = strcat("M",num2str(data{k,"M"}),"N",num2str(data{k,"N"}),"D",num2str(data{k,"D"}));
    temp = struct();
    temp.Time = data{k,"Tempo"}{1};
    temp.RunIn = data{k,"RunIn"}{1};
    temp.N_ensaio = data{k,"N_ensaio"}{1};
    temp.Unidade = data{k,"Unidade"}{1};
    temp.(name) = data{k,"Results"}.(model);
    temp = struct2table(temp);
    
    un_val = unique(temp(:,{'Unidade','N_ensaio'}));

    tabTemp = temp([],:);

    for k2 = 1:height(un_val)
        tempEnsaio = innerjoin(un_val(k2,:),temp,"Keys",{'N_ensaio','Unidade'});
        tempEnsaio = sortrows(tempEnsaio,{'Unidade','N_ensaio','Time'});
        runinMean = movmean(tempEnsaio.(name),[mvng_avg-1,0],"Endpoints","discard");
        tempEnsaio = tempEnsaio(mvng_avg:end,:);
        tempEnsaio.(name) = runinMean;

        tabTemp = [tabTemp;tempEnsaio];
    end
    
    temp = tabTemp;

    if ft
        tab = temp;
        ft = false;
    else
        tab = outerjoin(tab,temp,"MergeKeys",true);
    end

end

tableData = sortrows(tab,{'Unidade','N_ensaio','Time'});

end