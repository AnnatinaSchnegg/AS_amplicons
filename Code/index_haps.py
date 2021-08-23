#Module for assigning haplotypes to single cell amplicon data

#Authors:
#	Julie Thoms
#	Annatina Schnegg-Kaufmann
#	Fabio Zanini

#Version 1
#12th August 2021

#Import dependent modules
import os
import sys
import math
import numpy as np
import numpy.linalg
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import ternary
from scipy.stats import multinomial

#Define functions

def about():

	'''
	The functions in this module are used to assign haplotypes to single cells genotyped with PCR amplicons.

    The callable functions are;
	    data_retrieval(sourcefile, metadata, pt_id)
	    call_haps(data, pt_id, haps, reads,  cutoff)
		plot_hap_dist_sort_type(data, pt_id, haps, reads, cutoff, save = False)
		plot_index_heatmap(data, title, haps, reads, cutoff, save = False)
        calc_scVAF(data, pt_init, reads)
        calc_scVAF_binary_per_cell(data, pt_init, reads, cutoff)
	
	'''
	print('This module contains functions to assign haplotypes. For more info type index_haps.about?')


def data_retrieval(sourcefile, metadata, pt_id):

    '''
    This function reads data for a single patient from a master spreadsheet with amplicon data for all plates/patients.
    Input is the sourcefile with the readcounts (allele_counts_anon), and a metadata file which contains cell type for each plate (Amplicon_metadata_fixed_anon.xlsx).
    The function returns a dataframe containing just the data for the specified patient, ready to merge with index data and then plot.
    '''

    df = pd.read_csv(sourcefile, header = [0,1,2], index_col = 0, sep='\t')
    df = df.stack([0,1,2])
    df = df.reorder_levels([1,0,2,3])
    df = df.to_frame()  #puts everything back in a dataframe
    df.columns = ['Reads']
    df['Plate'] = df.index.get_level_values(0)  #These lines send indexes to columns
    df['Well'] = df.index.get_level_values(1)
    df['Amplicon'] = df.index.get_level_values(2)
    df['Genotype'] = df.index.get_level_values(3)
    df[['Patient', 'one', 'two']] = df['Amplicon'].str.split('_', expand = True)
    df = df.drop(columns = ['one', 'two'])

    #Import information about plate cell type and patient
    key = pd.read_excel(metadata, sheet_name = 'PlateID') #should this be an input? also in next fucntion
    key = key.drop(['Cell Origin', 'Plate Nr', 'Plate Name','Nr of cells'], axis=1)
    key.rename(columns = {'Comments2':'Plate'}, inplace = True)
    key.rename(columns = {'Cell-group':'Celltype'}, inplace = True)

    #Make a dictionary to associate plates with patients and plate with cell type
    plate_pt_dict = dict(zip(key.Plate, key.Patient))
    plate_cell_dict = dict(zip(key.Plate, key.Celltype))

    #Now just look at data from selected patient, and apply filters to identify cells with enough reads/amplicon
    pt_allele_plate = df.loc[df['Patient'].isin([pt_id])] 
    pt_allele_plate = pt_allele_plate.drop(columns = 'Patient') #Drop the Patient ID column and other unwanted cols
    pt_allele_plate['Cell_type'] = pt_allele_plate['Plate'].replace(plate_cell_dict)
    pt_allele_plate['Plate_Well'] = pt_allele_plate['Plate'].astype(str) + '_' + pt_allele_plate['Well'].astype(str)

    return pt_allele_plate

def call_haps(data, pt_id, haps, reads,  cutoff):
    
    cond = f'{pt_id}_{haps}'
    print(cond)
    
    if cond == 'JP001_2':
        cols = ['JP001_RUNX1_c', 'JP001_RUNX1_g']
        allcols = ['JP001_RUNX1_c','JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_c','JP001_TET2b_g']
    elif cond == 'JP001_3':
        cols = ['JP001_RUNX1_g', 'JP001_SRSF2', 'JP001_TET2a']
        allcols = ['JP001_RUNX1_c','JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_c','JP001_TET2b_g']
    elif cond == 'JP001_4':
        cols = ['JP001_RUNX1_g', 'JP001_SRSF2', 'JP001_TET2a', 'JP001_TET2b_g']
        allcols = ['JP001_RUNX1_c','JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_c','JP001_TET2b_g']
    elif cond == 'PD7153_3':
        cols = ['PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b']
        allcols = ['PD7153_CUX1', 'PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b', 'PD7153_TGFB3_c', 'PD7153_TGFB3_g']
    elif cond == 'PD7153_4':   
        cols = ['PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b',  'PD7153_TGFB3_g']
        allcols = ['PD7153_CUX1', 'PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b', 'PD7153_TGFB3_c', 'PD7153_TGFB3_g']
    elif cond == 'PD7151_2': 
        cols = ['PD7151_TET2a', 'PD7151_TET2b']
        allcols = ['PD7151_TET2a', 'PD7151_TET2b']
    else:
        print('For JP001 enter 2/3/4 haplotypes, for PD7153 enter 3/4 haplotypes, for PD7151 enter 2 haplotypes')
    
    #Import information about plate cell type and patient
    key = pd.read_excel('../Data/Amp_data/Amplicon_metadata_fixed.xlsx', sheet_name = 'PlateID')
    key = key.drop(['Cell Origin', 'Plate Nr', 'Plate Name','Nr of cells', 'fcs-fle' ], axis=1)
    key.rename(columns = {'Comments2':'Plate'}, inplace = True)
    key.rename(columns = {'Cell-group':'Celltype'}, inplace = True)
    
    #Make a dictionary to associate plates with patients and plate with cell type
    plate_pt_dict = dict(zip(key.Plate, key.Patient))
    plate_cell_dict = dict(zip(key.Plate, key.Celltype))
    
    #Group the data and apply filters
    df = data.copy()
    df = df.groupby(['Plate', 'Well', 'Amplicon']).sum().unstack()
    df.columns = allcols
    
    df = df.loc[(df[cols] >= reads).all(axis=1)] #df1 contains just the rows with cells we want - use this to create a filter or key
    df['Plate'] = df.index.get_level_values(0)  #These lines send indexes to columns
    df['Well'] = df.index.get_level_values(1)
    df['Plate_Well'] = df['Plate'].astype(str) + '_' + df['Well'].astype(str)
    wells = df['Plate_Well'].drop_duplicates().to_list() 
    print(f'Cells with {reads} reads for {haps} genes = ', len(wells))

    df2 = data.copy()
    df2 = df2[df2['Plate_Well'].isin(wells)]
    df2 = df2[df2['Amplicon'].isin(cols)]
    
    #Calculate the allele frequency
    df2 = df2.iloc[:, 0:1].unstack(level = 3)
    df2['Total'] = df2.iloc[: , 0] + df2.iloc[: , 1]
    df2['Mut_freq'] = df2.iloc[:, 0]/df2['Total']
    
    #Assign Wt or MT to each allele
    df2 = df2.drop(columns = ['Reads', 'Total'])

    conditions = [(df2['Mut_freq'] <= cutoff), (df2['Mut_freq']) > cutoff ]
    values = ['w', 'm']
    df2['Genotype'] = np.select(conditions, values)
    df2 = df2.drop(columns = ['Mut_freq']).unstack(2)
    df2.columns = cols
    
    if 'JP001_RUNX1_g' in df2.columns:
        df2.loc[:,'JP001_RUNX1_g'].replace({'w':'R','m':'r' }, inplace = True)
        
    if 'JP001_SRSF2' in df2.columns:  
        df2.loc[:,'JP001_SRSF2'].replace({'w':'S','m':'s' }, inplace = True)
        
    if 'JP001_TET2a' in df2.columns:     
        df2.loc[:,'JP001_TET2a'].replace({'w':'A','m':'a' }, inplace = True)
        
    if 'JP001_TET2b_g' in df2.columns:
        df2.loc[:,'JP001_TET2b_g'].replace({'w':'B','m':'b' }, inplace = True)
        
    if 'JP001_RUNX1_c' in df2.columns:   
        df2.loc[:,'JP001_RUNX1_c'].replace({'w':'C','m':'c' }, inplace = True)
        
    if 'PD7153_SRSF2' in df2.columns:   
        df2.loc[:,'PD7153_SRSF2'].replace({'w':'S','m':'s' }, inplace = True)
        
    if 'PD7153_TET2a' in df2.columns:   
        df2.loc[:,'PD7153_TET2a'].replace({'w':'A','m':'a' }, inplace = True)
        
    if 'PD7153_TET2b' in df2.columns:   
        df2.loc[:,'PD7153_TET2b'].replace({'w':'B','m':'b' }, inplace = True)
        
    if 'PD7153_TGFB3_g' in df2.columns:   
        df2.loc[:,'PD7153_TGFB3_g'].replace({'w':'T','m':'t' }, inplace = True)
        
    if 'PD7151_TET2a' in df2.columns:   
        df2.loc[:,'PD7151_TET2a'].replace({'w':'A','m':'a' }, inplace = True)
        
    if 'PD7151_TET2b' in df2.columns:   
        df2.loc[:,'PD7151_TET2b'].replace({'w':'B','m':'b' }, inplace = True)

    
    df2['Haplotype'] = 'x'

    for idx, row in df2.iterrows():
        
        if cond == 'JP001_3':
            a = row['JP001_SRSF2'] + row['JP001_TET2a'] + row['JP001_RUNX1_g']
        elif cond == 'JP001_4':
            a = row['JP001_SRSF2'] + row['JP001_TET2a'] + row['JP001_TET2b_g'] + row['JP001_RUNX1_g']
        elif cond == 'JP001_2':
            a = row['JP001_RUNX1_c'] + row['JP001_RUNX1_g']   
        elif cond == 'PD7153_3':
            a = row['PD7153_TET2b'] + row['PD7153_TET2a'] + row['PD7153_SRSF2']
        elif cond == 'PD7153_4':
            a = row['PD7153_TET2b'] + row['PD7153_TET2a'] + row['PD7153_SRSF2'] + row['PD7153_TGFB3_g']
        elif cond == 'PD7151_2':
            a = row['PD7151_TET2b'] + row['PD7151_TET2a']
        
        row['Haplotype'] = row['Haplotype'].replace('x', a)   

    df2['Sort_cell_type'] = df2.index.get_level_values(0)
    df2['Sort_cell_type'] = df2['Sort_cell_type'].replace(plate_cell_dict)
    df2['Plate'] = df2.index.get_level_values(0)
    df2['Well'] = df2.index.get_level_values(1)
    df2['Plate_Well'] = df2['Plate'].astype(str) + '_' + df2['Well'].astype(str)
    df2 = df2.drop(columns = cols)
    df2 = df2.drop(columns = ['Plate', 'Well'])
    
    return df2


def plot_hap_dist_sort_type(data, pt_id, haps, reads, cutoff, save = False): #plot based on cell type (ned to merge output from above)
      
    #rename the input data and work out how many haplotypes it has
    df3 = data.copy()
    haps = len(df3.iloc[0,0])
    cond = f'{pt_id}_{haps}'
    sortcells = df3['Sort_cell_type'].drop_duplicates().to_list()
    cellnumb = len(sortcells)
    plotlen = int(math.ceil((cellnumb +1)/2))
    
    #Plot two haplotype data for 3 gene 100 amplicon set - second method to add colour for each haplotype
    fig, axes = plt.subplots(plotlen,2, figsize = (16,plotlen*2))
    fig.subplots_adjust(hspace = 1.2, wspace=.3)
    ax = axes.ravel()
    count = 0

    c = df3['Haplotype'].value_counts().rename_axis('hap').reset_index(name='counts')

    #set up correct variables for the number of input haplotypes
    
    if cond == 'JP001_2':
        hap_poss = ['CR', 'Cr', 'cR', 'cr']
        
    elif cond == 'JP001_3':
        hap_poss = ['SAR', 'SAr', 'SaR', 'Sar', 'sAR', 'sAr', 'saR', 'sar']

    elif cond == 'JP001_4':
        hap_poss = ['SABR', 'SABr', 'SAbR', 'SAbr', 'SaBR', 'SaBr', 'SabR', 'Sabr', 'sABR', 'sABr', 'sAbR', 'sAbr', 'saBR', 'saBr', 'sabR', 'sabr']
      
    elif cond == 'PD7151_2':
        hap_poss = ['BA', 'Ba', 'bA', 'ba']

    elif cond == 'PD7153_3':
        hap_poss = ['BAS', 'BAs', 'BaS', 'Bas', 'bAS', 'bAs', 'baS', 'bas']

    elif cond == 'PD7153_4':
        hap_poss = ['BAST', 'BASt', 'BAsT', 'BAst', 'BaST', 'BaSt', 'BasT', 'Bast', 'bAST', 'bASt', 'bAsT', 'bAst', 'baST', 'baSt', 'basT', 'bast']

    num_col = len(hap_poss)
    cols = sns.color_palette("husl", num_col)     
    color = dict(zip(hap_poss, cols))
    hap_order = {}
    for i, j in enumerate(hap_poss):
        hap_order[1] = j
        
    #if any haplotype is not present, add it into the frame with freq 0 

    for h in hap_poss:
        if h not in str(c['hap']):  #for some reason this needs to be called as a string, wasn't needed outside function
            dfh = pd.DataFrame([[h, 0]], columns= ['hap', 'counts'])
            c = c.append(dfh)
            
    c['order'] = c['hap']
    c = c.replace({'order': hap_order})
    c = c.sort_values(by=['order'])
    d = c['counts'].sum()
    c['proportion'] = c['counts']/d

    sns.barplot(x='hap', y='counts', data = c, palette = color, ax = ax[0], ci = None) #fro scatter add  hue = 'hap'
    ax[0].set_title('All cells') 
    ax[0].set_ylabel('Number of cells', fontsize = 11)
    ax[0].set_xlabel('Haplotype', fontsize = 11)
    ax[0].tick_params(axis='x', labelrotation = 90)

    
    
    for cell in sortcells:
        count += 1
    
        if df3.loc[df3['Sort_cell_type'].isin([cell])].empty == False:

            a = df3.loc[df3['Sort_cell_type'].isin([cell])]['Haplotype'].value_counts().rename_axis('hap').reset_index(name='counts')

            #if any haplotype is not present, add it into the frame with freq 0 hap_3gene_poss has the possibilities
            for h in hap_poss:
                if h not in str(a['hap']):  #for some reason this needs to be called as a string, wasn't needed outside function
                    dfh = pd.DataFrame([[h, 0]], columns= ['hap', 'counts'])
                    a = a.append(dfh)    

            a['order'] = a['hap']
            a = a.replace({'order': hap_order})
            a = a.sort_values(by=['order'])
            b = a['counts'].sum()
            a['proportion'] = a['counts']/b

            sns.barplot(x='hap', y='counts', data = a, palette = color,  ax = ax[count], ci = None) #for scatter add  hue = 'hap'
            ax[count].set_title(str(cell)) 
            ax[count].set_ylabel('Number of cells', fontsize = 11)
            ax[count].set_xlabel('Haplotype', fontsize = 11)
            ax[count].tick_params(axis='x', labelrotation = 90)

        else:
            continue
            
    if save == True:
        fig.savefig(f'../Results/Haplotypes/{pt_id}_{haps}_{reads}_{cutoff}_haplotype_by_sortcell.png',dpi=300)

def plot_index_heatmap(data, title, haps, reads, cutoff, save = False):
    df = data.copy()
    a = df.groupby(['Haplotype', 'celltype']).size().unstack(fill_value = 0)
    alltypes = ['HSC','MPP','HEALTHY_SC','CMP',  'GMP','GMP2', 'MEP', 'MDS_SC',  'NE', 'Mono','nBC', 'unassigned']
    col_order = {}            
    for i, typ in enumerate(alltypes):
         col_order[typ] = i

    a = a.T
    a['ct'] = a.index.get_level_values(0)
    a = a.replace({'ct': col_order})
    a = a.sort_values(by=['ct'])
    a = a.drop(columns = ['ct'])
    a = a.T
    b = a.copy()
    a = a * 100 /a.sum(axis = 0)

    fig, ax = plt.subplots(figsize = (16, 6))
    sns.heatmap(data = a, ax = ax, robust = True)
    ax.tick_params(axis='y', labelrotation = 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)  
    
    x = df.groupby(['Haplotype', 'celltype']).size().unstack(fill_value = 0)
    y = df.groupby(['Haplotype', 'celltype']).size().unstack(fill_value = 0)
    x = x.sum(axis = 0)
    x = x.to_frame()
    x['order'] = x.index.get_level_values(0)
    x = x.replace({'order': col_order})
    x = x.sort_values(by=['order'])
    x = x.drop(columns = ['order'])
    x['ct'] = x.index.get_level_values(0)
    x.columns = ['number', 'ct']

    
    fig2, ax2 = plt.subplots(figsize = (12.8, 1))
    sns.scatterplot(x = 'ct', y = 'number', data = x, ax = ax2, s = 100, color = 'green')
    ax2.tick_params(axis='x', labelrotation = 90)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.axhline(10, ls = '--', c = 'gray')
    ax2.axhline(100, ls = '--', c = 'gray')
    ax2.set_ylim(1,1001)
    ax2.set_yticks([1,10, 100, 1000])
    ax2.set_yticklabels(['1','10', '100','1000]'])    
    ax2.set_yscale('log') #use this only with scatterplot
    ax2.set_title('Total number of cells for each type')  
    
    if save == True:
        fig.savefig(f'../Results/Haplotypes/{pt_id}_{haps}_{reads}_{cutoff}_haplotype_by_indexcell.png',dpi=300)
    
    return b


def calc_scVAF(data, pt_init, reads):
    
    '''
    This function takes amplicon read counts for mt and wt and calculates the proportion of mutated alleles in each cell that meets the specified read count.
    Not sure this is the best method though, the second function calc_scVAF_binary_per_cell() is aligned to method for calling haps.
    All CD34pos cells are treated as a single sample.
    The function returns a plot and data suitable for plotting (eg/ after merging all samples).
    '''
    
    cond = pt_init
    print(cond)
    
    if cond == 'JP001':  #cols is the columns actually being used, can be easily tweaked
        cols = ['JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a']
        #cols = ['JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_g']
        allcols = ['JP001_RUNX1_c','JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_c','JP001_TET2b_g']
    elif cond == 'PD7153':
        cols = ['PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b']
        allcols = ['PD7153_CUX1', 'PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b', 'PD7153_TGFB3_c', 'PD7153_TGFB3_g']
    elif cond == 'PD7151': 
        cols = ['PD7151_TET2a', 'PD7151_TET2b']
        allcols = ['PD7151_TET2a', 'PD7151_TET2b']
        
    else:
        print('Enter JP001,  PD7153, or PD7151 as pt_id')
    
    #Import information about plate cell type and patient
    key = pd.read_excel('../Data/Amp_data/Amplicon_metadata_fixed_anon.xlsx', sheet_name = 'PlateID')
    key = key.drop(['Cell Origin', 'Plate Nr', 'Plate Name','Nr of cells'], axis=1)
    key.rename(columns = {'Comments2':'Plate'}, inplace = True)
    key.rename(columns = {'Cell-group':'Celltype'}, inplace = True)
    
    #Make a dictionary to associate plates with patients and plate with cell type
    plate_pt_dict = dict(zip(key.Plate, key.Patient))
    plate_cell_dict = dict(zip(key.Plate, key.Celltype))
    
    #Group the data and apply filters
    df = data.copy()
    df = df.groupby(['Plate', 'Well', 'Amplicon']).sum().unstack()
    df.columns = allcols
    
    df = df.loc[(df[cols] >= reads).all(axis=1)] #df1 contains just the rows with cells we want - use this to create a filter or key
    df['Plate'] = df.index.get_level_values(0)  #These lines send indexes to columns
    df['Well'] = df.index.get_level_values(1)
    df['Plate_Well'] = df['Plate'].astype(str) + '_' + df['Well'].astype(str)
    wells = df['Plate_Well'].drop_duplicates().to_list() 
    print(f'Cells with {reads} reads  = ', len(wells))

    df2 = data.copy()
    df2 = df2[df2['Plate_Well'].isin(wells)]
    df2 = df2[df2['Amplicon'].isin(cols)]
    
    #Calculate the allele frequency
    df2 = df2.iloc[:, 0:1].unstack(level = 3)
    df2['Total'] = df2.iloc[: , 0] + df2.iloc[: , 1]
    df2['Mut_freq'] = df2.iloc[:, 0]/df2['Total']
    

    #Assign Wt or MT to each allele
    df2 = df2.drop(columns = ['Reads', 'Total'])
    df2 = df2.unstack(2)
    df2.columns = cols


    df2['Sort_cell_type'] = df2.index.get_level_values(0)
    df2['Sort_cell_type'] = df2['Sort_cell_type'].replace(plate_cell_dict)
    df2['Plate'] = df2.index.get_level_values(0)
    df2['Well'] = df2.index.get_level_values(1)
    df2['Plate_Well'] = df2['Plate'].astype(str) + '_' + df2['Well'].astype(str)
    df2 = df2.drop(columns = ['Plate', 'Well'])
    
    rename = {'CD34+halfCD38-': 'CD34', 'CD34+/38-':'CD34', 'CD34+':'CD34'}
    df2['Sort_cell_type'].replace(rename, inplace = True)
    df2.sort_values(by=['Sort_cell_type'], inplace = True)
    
    x = df2.copy().groupby(by = 'Sort_cell_type').mean()
    x = x.unstack().to_frame()
    x['celltype'] = x.index.get_level_values(1)
    x['Amplicon'] = x.index.get_level_values(0)
    co = ['VAF', 'sort_celltype', 'Amplicon']
    x.columns = co
    x.sort_values(by=['sort_celltype'], inplace = True)
    
    all_amps = ['JP001_SRSF2',
            'PD7153_TET2b',
            'JP001_TET2a',
            'PD7153_TET2a', 
            'PD7151_TET2b',
            'JP001_TET2b_g', 
            'PD7153_SRSF2', 
            'JP001_RUNX1_g',
            'PD7153_CUX1',
            'PD7151_TET2a'
           ]
    colors = sns.color_palette('husl', n_colors = len(all_amps))
    allVAFcols = dict(zip(all_amps, colors))

    fig, ax = plt.subplots(figsize = (2.5,4))
    sns.scatterplot(x = 'sort_celltype', y = 'VAF', data = x, s = 100,  hue = 'Amplicon', palette = allVAFcols, alpha = 0.5, ax = ax)
    ax.legend(loc = 'upper left', bbox_to_anchor = [1,1], title = 'scVAFs by read frequency')
    ax.set_ylim(0,0.6)
    ax.axhline(0.1, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.2, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.3, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.4, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.5, ls = '--', c = 'silver', zorder = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation = 90)
    
    return x

def calc_scVAF_binary_per_cell(data, pt_init, reads, cutoff):
    
    '''
    This function takes amplicon read counts for mt and wt and calculates whether a given cell does (1) or does not (0) contain a mutant allele, based on the desired read count and cutoff values.
    All CD34pos cells are treated as a single sample.
    The function returns a plot and data suitable for plotting (eg/ after merging all samples).
    '''
    
    cond = pt_init
    print(cond)
    
    if cond == 'JP001':  #cols is the columns actually being used, can be easily tweaked
        #cols = ['JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a']
        cols = ['JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_g']
        allcols = ['JP001_RUNX1_c','JP001_RUNX1_g','JP001_SRSF2','JP001_TET2a','JP001_TET2b_c','JP001_TET2b_g']
    elif cond == 'PD7153':
        cols = ['PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b']
        allcols = ['PD7153_CUX1', 'PD7153_SRSF2', 'PD7153_TET2a', 'PD7153_TET2b', 'PD7153_TGFB3_c', 'PD7153_TGFB3_g']
    elif cond == 'PD7151': 
        cols = ['PD7151_TET2a', 'PD7151_TET2b']
        allcols = ['PD7151_TET2a', 'PD7151_TET2b']
        
    else:
        print('Enter JP001,  PD7153, or PD7151 as pt_id')
    
    #Import information about plate cell type and patient
    key = pd.read_excel('../Data/Amp_data/Amplicon_metadata_fixed_anon.xlsx', sheet_name = 'PlateID')
    key = key.drop(['Cell Origin', 'Plate Nr', 'Plate Name','Nr of cells'], axis=1)
    key.rename(columns = {'Comments2':'Plate'}, inplace = True)
    key.rename(columns = {'Cell-group':'Celltype'}, inplace = True)
    
    #Make a dictionary to associate plates with patients and plate with cell type
    plate_pt_dict = dict(zip(key.Plate, key.Patient))
    plate_cell_dict = dict(zip(key.Plate, key.Celltype))
    
    #Group the data and apply filters
    df = data.copy()
    df = df.groupby(['Plate', 'Well', 'Amplicon']).sum().unstack()
    df.columns = allcols
    
    df = df.loc[(df[cols] >= reads).all(axis=1)] #df1 contains just the rows with cells we want - use this to create a filter or key
    df['Plate'] = df.index.get_level_values(0)  #These lines send indexes to columns
    df['Well'] = df.index.get_level_values(1)
    df['Plate_Well'] = df['Plate'].astype(str) + '_' + df['Well'].astype(str)
    wells = df['Plate_Well'].drop_duplicates().to_list() 
    print(f'Cells with {reads} reads  = ', len(wells))

    df2 = data.copy()
    df2 = df2[df2['Plate_Well'].isin(wells)]
    df2 = df2[df2['Amplicon'].isin(cols)]
    
    #Calculate the allele frequency
    df2 = df2.iloc[:, 0:1].unstack(level = 3)
    df2['Total'] = df2.iloc[: , 0] + df2.iloc[: , 1]
    df2['Mut_freq'] = df2.iloc[:, 0]/df2['Total']
    
    #Assign Wt or MT to each allele
    df2 = df2.drop(columns = ['Reads', 'Total'])

    conditions = [(df2['Mut_freq'] <= cutoff), (df2['Mut_freq']) > cutoff ]
    values = [0, 1]
    df2['Genotype'] = np.select(conditions, values)
    df2 = df2.drop(columns = ['Mut_freq']).unstack(2)
    df2.columns = cols


    df2['Sort_cell_type'] = df2.index.get_level_values(0)
    df2['Sort_cell_type'] = df2['Sort_cell_type'].replace(plate_cell_dict)
    df2['Plate'] = df2.index.get_level_values(0)
    df2['Well'] = df2.index.get_level_values(1)
    df2['Plate_Well'] = df2['Plate'].astype(str) + '_' + df2['Well'].astype(str)
    df2 = df2.drop(columns = ['Plate', 'Well'])
    
    rename = {'CD34+halfCD38-': 'CD34', 'CD34+/38-':'CD34', 'CD34+':'CD34'}
    df2['Sort_cell_type'].replace(rename, inplace = True)
    df2.sort_values(by=['Sort_cell_type'], inplace = True)
    
    cells = df2['Sort_cell_type'].drop_duplicates().to_list()
    heads = ['Sort_cell_type', 'Amplicon', 'n_mut', 'vaf', 'total']
    data = pd.DataFrame(columns = heads)
    for c in cells:
        df4 = df2.loc[df2['Sort_cell_type'].isin([c])]
        data2 = {}
        for co in cols:
            tot = len(df4[co])
            n_mut = df4[co].sum()     
            vaf = n_mut/tot
            sers = [c, co, n_mut, vaf, tot]
            df5 = pd.Series(data=sers, index = heads)
            data = data.append(df5, ignore_index = True)

    data.sort_values(by=['Sort_cell_type'], inplace = True)
    
    all_amps = ['JP001_SRSF2',
            'PD7153_TET2b',
            'JP001_TET2a',
            'PD7153_TET2a', 
            'PD7151_TET2b',
            'JP001_TET2b_g', 
            'PD7153_SRSF2', 
            'JP001_RUNX1_g',
            'PD7153_CUX1',
            'PD7151_TET2a'
           ]
    colors = sns.color_palette('husl', n_colors = len(all_amps))
    allVAFcols = dict(zip(all_amps, colors))
    
    fig, ax = plt.subplots(figsize = (2.5,4))
    sns.scatterplot(x = 'Sort_cell_type', y = 'vaf', data = data, s = 100,  hue = 'Amplicon', palette = allVAFcols, alpha = 0.5, ax = ax)
    ax.legend(loc = 'upper left', bbox_to_anchor = [1,1], title = 'scVAFs by wt/mt status')
    ax.set_ylim(0,1.1)
    #ax.axhline(0.1, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.2, ls = '--', c = 'silver', zorder = 0)
    #ax.axhline(0.3, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.4, ls = '--', c = 'silver', zorder = 0)
    #ax.axhline(0.5, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.6, ls = '--', c = 'silver', zorder = 0)
    #ax.axhline(0.7, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(0.8, ls = '--', c = 'silver', zorder = 0)
    #ax.axhline(0.9, ls = '--', c = 'silver', zorder = 0)
    ax.axhline(1.0, ls = '--', c = 'silver', zorder = 0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation = 90)
    
    
    return data



if __name__ == "__main__":

	access_module()