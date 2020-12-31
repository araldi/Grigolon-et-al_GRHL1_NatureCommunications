#load the files bgen 
import pandas as pd 
import numpy as np
from bgen_reader import read_bgen

bgen2 = read_bgen("/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated.bgen" )
df = pd.DataFrame()
variant = bgen2["variants"].compute()
column = variant["rsid"]
for i in range(len(column)):
    geno = bgen2['genotype'][i].compute()
    alleles = bgen2['variants'].compute()['allele_ids'][i]
    thiscol = np.empty((geno['probs'].shape[0],), dtype='U3')
    masks = geno['probs'] >= 0.9
    thiscol[masks[:, 0]] = alleles[0]  + ' ' +  alleles[0]
    thiscol[masks[:, 1]] = alleles[0]  + ' ' +  alleles[-1]
    thiscol[masks[:, 2]] = alleles[-1]  + ' ' + alleles[-1]
    df[column[i]] = thiscol
df['eid'] = bgen2["samples"].astype(int)
df.to_csv('/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated.csv')

original_file = '/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/ukb37820_no_genotype.csv'
df_original = pd.read_csv(original_file, encoding='latin-1')

df_newSNPs = pd.merge(df, df_original, how="outer", on="eid")
df_newSNPs.to_csv('/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated_merged_ukb37820.csv')

#prepare dataset
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import datetime
import calendar
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import traceback

sns.set_style("ticks")


date = datetime.now().strftime('%Y.%m.%d')

gene = "gene_of_interest"

file_stats = '/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/snp-stats_annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated.txt'
file_dataset = '/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated_merged_ukb37820.csv'
file_dataset_survival = '/cluster/home/earaldi/scratch/Data/UK_BIOBANK_BULK/Genotypes/Imputed/extracted/annotated_gene_of_interest_cis-eQTLs_SNPs_upregulated_merged_ukb37820.csv'


#load stats and create MAF dictionary
stats = pd.read_csv(file_stats, skiprows=8, sep=' ')
stats.drop(stats.tail(1).index,inplace=True)

SNP_MAF_dic = {}
homoWT = {}
HET = {}
homoMUT = {}

for SNP in stats.itertuples():
    SNP_MAF_dic[SNP.rsid] = SNP.alleleB_frequency
    homoWT[SNP.rsid] = "%s %s" %(SNP.alleleA, SNP.alleleA)
    HET[SNP.rsid] = "%s %s" %(SNP.alleleB, SNP.alleleA)
    homoMUT[SNP.rsid] = "%s %s" %(SNP.alleleB, SNP.alleleB)

# df_analysis = pd.read_csv(file_dataset) ##if starting from cluster/scratch
df_analysis = df_newSNPs

df_analysis = df_analysis.rename(columns ={ 'eid':'PatientID',
                                           '21022-0.0': 'AgeRecruit',
                                           '53-0.0' : 'DateRecruit',
                                           '34-0.0':'YearBirth',
                                           '52-0.0' : 'MonthBirth',
                                           '40007-0.0': 'AgeDeath',
                                           '40000-0.0': 'DateDeath',
                                           '31-0.0':'Sex'})

#calculate precise age between DOB and Daterecruit, assuming that the DOB is on 15th of every month
# if this is not the case, use the age at recruitment

df_analysis['DOB'] = pd.to_datetime((df_analysis.YearBirth*10000+df_analysis.MonthBirth*100+15).apply(str),format='%Y%m%d')
df_analysis['DateDeath'] = pd.to_datetime(df_analysis.DateDeath.apply(str))
df_analysis['EstimAgeRecruit'] = pd.to_datetime(df_analysis['DateRecruit'])- df_analysis['DOB']
df_analysis['EstimAgeRecruit'] = df_analysis['EstimAgeRecruit']/np.timedelta64(1,'Y')

#for 4520 volunteers, the birthday is between the beginning of the month 
#and the recruitment date, before the 15th of the month: the estimated age of recruiment is smaller than 
#the declared Age at recruitment. For those volunteers, round up the estimated age at recruitment

myidx = df_analysis['EstimAgeRecruit'] - df_analysis['AgeRecruit'] < 0 
df_analysis["EstimAgeRecruit"][myidx] = np.ceil(df_analysis["EstimAgeRecruit"][myidx])
df_analysis['CurrentAgeAlive'] = np.nan
myidx = pd.isnull(df_analysis['AgeDeath'])
df_analysis['CurrentAgeAlive'][myidx] =  (pd.to_datetime((2018*1000+2*100+14),format='%Y%m%d') - df_analysis['DOB']) /np.timedelta64(1,'Y')


#data for the statsmodels.duration.hazard_regression.PHReg
#this is not tested yet

df_analysis['Observed_Statsmodels'] = 1
df_analysis['Duration_Statsmodels'] = pd.to_numeric((df_analysis['DateDeath']- df_analysis['DOB']).dt.days, downcast='integer')
myindx = pd.notnull(df_analysis['CurrentAgeAlive'])
df_analysis['Observed_Statsmodels'][myindx] = 0
df_analysis['Duration_Statsmodels'][myindx] = pd.to_numeric((pd.to_datetime((2018*1000+2*100+14),format='%Y%m%d') - df_analysis['DOB']).dt.days, downcast='integer')


PC = []
for i in range(1,21):
    PC.append("22009-0.%s" %i)
other_columns = ['Sex', 'Duration_Statsmodels', 'Observed_Statsmodels', 
                 'PatientID', 'AgeRecruit','DateRecruit',
                 'YearBirth', 'MonthBirth', 'AgeDeath', 'DateDeath']
SNPs = [i for i in stats['rsid']]

df_analysis = df_analysis[other_columns + SNPs + PC].copy()
df_analysis.to_csv(file_dataset_survival)

#SURVIVAL ANALYSIS
#SURVIVAL ANALYSIS
#SURVIVAL ANALYSIS
#SURVIVAL ANALYSIS
#SURVIVAL ANALYSIS


#test and run the survival analysis



import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import datetime
import calendar
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import traceback

gene_of_interest = "gene_of_interest_cis_upregulated"
results_directory = '/cluster/home/earaldi/Scripts/ristow_lab/gene_of_interest/Results'
output_file = 'survival_SNPs_%s' %gene_of_interest



cox_results = pd.DataFrame(columns = ['SNP (Genotype)', 'pvalue', 'logHR'])


#select the columns to use for the working dataframe  
SNPs = stats.rsid

#### CAREFUL HERE WITH MALES and FEMALES (imputed females=0)

## UPDATE according to 
females = 0
color ={0:'red', 1:'orange', 2:'magenta', 3:'blue', 4:'c', 5:'g'}
sex = {females:'females', 1:'males'}
sex_marker = {females: '+', 1 : 'o'}
funct = ['cumhaz', 'surv']
function = {'cumhaz': "Cumulative Hazard", 'surv' : 'Survival Probability'}

####
####include the levels
for SNP in SNPs:
    date = datetime.now().strftime('%Y.%m.%d')

    df_analysis2 = df_analysis.dropna(subset = ['Sex' , 'Duration_Statsmodels', SNP, '22009-0.1'])
    df_analysis2 = df_analysis2[df_analysis2[SNP] != '0 0']
    #df_analysis2['sex_geno'] = df_analysis2[SNP] + ' ' + df_analysis2['Sex'].map(sex)
    print("Before first try")  
 
    try:    
        #calculate pvalue using sex as stratum and genetic PC as covariates
        l = [homoWT[SNP] , HET[SNP] , homoMUT[SNP]]
        cox_pval = PHReg.from_formula("Duration_Statsmodels ~  C(%s, levels = %s)+ Q('22009-0.1')+ Q('22009-0.2')+ Q('22009-0.3')+ Q('22009-0.4')+ Q('22009-0.5')+ Q('22009-0.6')+ Q('22009-0.7')+ Q('22009-0.8')+ Q('22009-0.9')+ Q('22009-0.10') " %(SNP, l), 
                                 data= df_analysis2, status = df_analysis2["Observed_Statsmodels"], ties="breslow", strata = df_analysis2["Sex"])        
        print("Defined formula")
        results_pval = cox_pval.fit()
        print("Fitted formula")
        genotype0 = results_pval.summary().tables[1].index[0].split('[')[2].split('.')[1].split(']')[0]
        print(genotype0)
        pval0 = results_pval.pvalues[0]
        HR0 = results_pval.params[0]
        results0 = [SNP+'  (' +  genotype0+ ')', pval0, HR0]
        cox_results = cox_results.append(pd.Series(results0, index=['SNP (Genotype)', 'pvalue', 'logHR']), ignore_index=True)

        #check if second genotype is present for pvalue
        if SNP in results_pval.summary().tables[1].index[1]:
            genotype1 = results_pval.summary().tables[1].index[1].split('[')[2].split('.')[1].split(']')[0]
            pval1 = results_pval.pvalues[1]
            HR1 = results_pval.params[1]
            results1 = [SNP+'  (' + genotype1 + ')', pval1, HR1]
            cox_results = cox_results.append(pd.Series(results1, index=['SNP (Genotype)', 'pvalue', 'logHR']), ignore_index=True)

        #saves as it loops in case there the loop does not finish
        cox_results.to_csv('%s/%s_%s_partial.csv' %(results_directory, date, output_file))        

        try:
            cox_plot = PHReg.from_formula("Duration_Statsmodels ~  C(%s, levels = %s) + Sex " %(SNP, l), 
                                     data= df_analysis2, status = df_analysis2["Observed_Statsmodels"], ties="breslow")

            results_plot = cox_plot.fit()

            for f in funct:
                i=0
                plt.figure(figsize = [20,10])
                predictions = results_plot.predict(pred_type=f).predicted_values
                for s in sex:
                    for l in range(0,len(df_analysis2[SNP].value_counts())):
                        genot = df_analysis2[SNP].value_counts().index[l]
                        mask = ((df_analysis2[SNP]== genot) & (df_analysis2["Sex"]==s))
                        prediction_masked = ma.masked_array(predictions, ~mask)
                        duration_masked = ma.masked_array(df_analysis2["Duration_Statsmodels"], ~mask)
                        plt.scatter(duration_masked/365, prediction_masked, color = color[i] , marker= sex_marker[s], label="%s %s" %(genot, sex[s]))
                        #keep the color key identical among graphs
                        if l == 1:
                            if len(df_analysis2[SNP].value_counts()) <3:
                                i = i+2
                            else:
                                i = i+1
                        else:
                            i=i+1

                leg = plt.legend(numpoints=1, handletextpad=0.0001, fontsize=16)
                leg.draw_frame(False)
                plt.grid(True)
                plt.xlabel("Time (Years)")
                plt.ylabel("%s" %function[f])
                plt.suptitle("%s for %s" %(function[f], SNP), fontsize=20, fontweight="bold")
                if SNP in results_pval.summary().tables[1].index[1]:
                    plt.title("gene = %s, MAF=%.2e \n %s: pvalue = %.2e    logHR = %.4f \n %s: pvalue = %.4f    logHR = %.4f " % (gene, SNP_MAF_dic[SNP], genotype0, pval0,HR0, genotype1, pval1, HR1), fontsize=16)
                else:
                    plt.title("gene = %s, MAF=%.2e \n %s: pvalue = %.2e    logHR = %.4f " % (gene, SNP_MAF_dic[SNP], genotype0, pval0,HR0), fontsize=16)

                plt.savefig("%s/Graphs/%s_%s_%s_%s_ukb37820_Cox.png"  % (results_directory, date, output_file, SNP, f), dpi=300 )
                #plt.savefig("%s/Graphs/%s_%s_%s_%s_ukb37820_Cox.svg"  % (results_directory, date, output_file, SNP, f), dpi=300)
                #plt.savefig("%s/Graphs/%s_%s_%s_%s_ukb37820_Cox.pdf"  % (results_directory, date, output_file, SNP, f), dpi=300)
                plt.clf()
        except Exception as e:
            print("Gene: %s, SNP: %s, Exception interna: " % (gene, SNP), e)
            pass

    except Exception as e:
        print("Gene: %s, SNP: %s, Exception esterna: " % (gene, SNP), e)
        # traceback.print_exc()
        pass

cox_results['rsID'] = cox_results['SNP (Genotype)'].str.split(" ").str[0]
cox_results['MAF'] = cox_results['rsID'].map(SNP_MAF_dic)

cox_results.to_csv('%s/%s_%s_final.csv' %(results_directory, date, output_file)) 
