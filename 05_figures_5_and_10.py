

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pypsa
import os
import sys
import geopandas
import tables
import matplotlib
from operator import itemgetter, attrgetter
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
#from sklearn.preprocessing import Normalizer


# ## Helper functions

# In[2]:


## from RD_func.py import PCA, MAP
def PCA(X):
    """
    Input:
        - X: Matrix for performing PCA on
    Output:
        - eigen_values:         Eigen values for the input
        - eigen_vectors:        Eigen vectors for the input
        - variance_explained:   How much of the variance is explained by each principal component
        - norm_const:           Normalization constant
        - T:                    Principle component amplitudes (a_k from report)
    """
    X_avg = np.mean(X, axis=0)
    B = X.values - X_avg.values
    norm_const = (1 / (np.sqrt( np.sum( np.mean( ( (X - X_avg)**2 ), axis=0 ) ) ) ) )
    C = np.cov((B*norm_const).T)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    variance_explained = (eigen_values * 100 ) / eigen_values.sum()
    T = np.dot((B*norm_const), eigen_vectors)

    return (eigen_values, eigen_vectors, variance_explained, norm_const, T)


# In[3]:


#%% Season plot
def season_plot(T, time_index, file_name):
    """
    Parameters
    ----------
    T : Matrix
        Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
    data_index : panda index information
        index for a year (used by panda's dataframe')
    file_name: array of strings
        Name of the datafile there is worked with

    Returns
    -------
    Plot of seasonal distribution
    """
    T = pd.DataFrame(data=T,index=time_index)
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day

    # Upper left figure
    plt.figure(figsize=(16,10))
    plt.subplot(2,2,1)
    plt.plot(T_avg_hour[0],label='k=1')
    plt.plot(T_avg_hour[1],label='k=2')
    plt.plot(T_avg_hour[2],label='k=3')
    plt.xticks(ticks=range(0,24,2))
    plt.legend(loc='upper right',bbox_to_anchor=(1,1))
    plt.xlabel("Hours")
    plt.ylabel("a_k interday")
    plt.title("Hourly average for k-values for 2015 ")
    # Upper right figure
    x_ax = range(len(T_avg_day[0])) # X for year plot
    plt.subplot(2,2,2)
    plt.plot(x_ax,T_avg_day[0],label='k=1')
    plt.plot(x_ax,T_avg_day[1],label='k=2')
    plt.plot(x_ax,T_avg_day[2],label='k=3')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.xlabel("day")
    plt.ylabel("a_k seasonal")
    plt.title("daily average for k-values for 2015 ")
    # Lower left figure
    plt.subplot(2,2,3)
    plt.plot(T_avg_hour[3],label='k=4',color="c")
    plt.plot(T_avg_hour[4],label='k=5',color="m")
    plt.plot(T_avg_hour[5],label='k=6',color="y")
    plt.xticks(ticks=range(0,24,2))
    plt.legend(loc='upper right',bbox_to_anchor=(1,1))
    plt.xlabel("Hours")
    plt.ylabel("a_k interday")
    plt.title("Hourly average for k-values for 2015 ")
    # Lower right figure
    x_ax = range(len(T_avg_day[0])) # X for year plot
    plt.subplot(2,2,4)
    plt.plot(x_ax,T_avg_day[3],label='k=4',color="c")
    plt.plot(x_ax,T_avg_day[4],label='k=5',color="m")
    plt.plot(x_ax,T_avg_day[5],label='k=6',color="y")
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.xlabel("day")
    plt.ylabel("a_k seasonal")
    plt.title("daily average for k-values for 2015 ")
    # Figure title
    plt.suptitle(file_name,fontsize=20,x=.51,y=0.932) #,x=.51,y=1.07
    
    return plt.show(all)


# In[4]:


#%% Eigen values contribution plot
def eig_val_contribution_plot(lambda_collect_wmin, lambda_tot, file_name):
    """
    Parameters
    ----------
    lambda_collect_wmin : array
        Lambda array collected with those that are minus already included
    lambda_tot : value
        Sum of all lambda values
    file_name : array of strings
        Name of the datafile there is worked with

    Returns
    -------
    Plot of eigen value contribution
    """
    plt.figure(figsize=[14,16])
    for n in range(6):
        lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.subplot(3,2,n+1)
        plt.bar(lambda_collect_wmin.columns,lambda_collect_procent.values[0])
        plt.title('PC'+str(n+1)+': '+str(round(lambda_tot[n],3)))
        plt.ylabel('Influance [%]')
        plt.ylim([-50,125])
        plt.grid(axis='y',alpha=0.5)
        for k in range(10):
            if lambda_collect_procent.values[:,k] < 0:
                v = lambda_collect_procent.values[:,k] - 6.5
            else:
                v = lambda_collect_procent.values[:,k] + 2.5
            plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
    plt.suptitle(file_name,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07     
   
    return plt.show(all)


# In[5]:


#%% Principle component contribution plot
def PC_contribution_plot(PC_con, type_of_contribution):
    """
    Parameters
    ----------
    PC_con : list
        list of with sorted principle components contributions 
    type_of_contribution : string
        "mismatch" - used when wanting to plot for the mismatch case
        "respone" - used when wanting to plot for the response case
    
    Returns
    -------
    Plot of principle component distribution
    """
    plt.figure(figsize=(14,16))#,dpi=500)
    for i in range(6):
        if type_of_contribution == "mismatch":
            # y functions comulated
            wind_con_data  = PC_con[i][:,:1].sum(axis=1)
            solar_con_data = PC_con[i][:,:2].sum(axis=1)
            hydro_con_data = PC_con[i][:,:3].sum(axis=1)
            load_con_data  = PC_con[i][:,:4].sum(axis=1)
            gen_cov_data   = PC_con[i][:,:7].sum(axis=1)
            load_cov_data  = PC_con[i][:,8:10].sum(axis=1)
            # plot function
            plt.subplot(3,2,i+1)
            # Plot lines
            plt.plot(wind_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(solar_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(hydro_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(load_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(gen_cov_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(load_cov_data,color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(7), np.zeros(7), wind_con_data,
                             label='Wind',
                             color='cornflowerblue') # Because it is a beutiful color
            plt.fill_between(range(7), wind_con_data, solar_con_data,
                             label='Solar',
                             color='yellow')
            plt.fill_between(range(7), solar_con_data, hydro_con_data,
                             label='Hydro',
                             color='darkslateblue')
            plt.fill_between(range(7), hydro_con_data, load_con_data,
                             label='Load',
                             color='slategray')
            plt.fill_between(range(7), load_con_data, gen_cov_data,
                             label='Generator\ncovariance',
                             color='brown',
                             alpha=0.5)
            plt.fill_between(range(7), load_cov_data, np.zeros(7),
                             label='Load\ncovariance',
                             color='orange',
                             alpha=0.5)
            # y/x-axis and title
            #plt.legend(bbox_to_anchor = (1,1))
            plt.ylabel('$\lambda_k$')
            plt.xticks(np.arange(0,7),['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
            plt.title('Principle component '+str(i+1))
            if i == 4: # Create legend of figure 4 (lower left)
                plt.legend(loc = 'center', bbox_to_anchor = (1.1,-0.17), ncol = 6, 
                           fontsize = 'large', framealpha = 1, columnspacing = 2.5)
        
        # Principle components for nodal response
        elif type_of_contribution == "response": 
            # y functions comulated
            backup_con_data  = PC_con[i][:,:1].sum(axis=1)
            inport_export_con_con_data = PC_con[i][:,:2].sum(axis=1)
            storage_con_data = PC_con[i][:,:3].sum(axis=1)
            #backup_inport_cov_data  = PC_con[i][:,:4].sum(axis=1)
            #backup_store_cov_data   = PC_con[i][:,:5].sum(axis=1)
            inport_store_cov_data   = PC_con[i][:,:6].sum(axis=1)
            # plot function
            plt.subplot(3,2,i+1)
            # Plot lines
            plt.plot(backup_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(inport_export_con_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(storage_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(inport_store_cov_data,color='k',alpha=1,linewidth=0.5)
            plt.fill_between(range(7), np.zeros(7), backup_con_data,
                             label='backup',
                             color='cornflowerblue') # Because it is a beutiful color
            plt.fill_between(range(7), backup_con_data, inport_export_con_con_data,
                             label='import & export',
                             color='yellow')
            plt.fill_between(range(7), inport_export_con_con_data, storage_con_data,
                             label='storage',
                             color='darkslateblue')
            plt.fill_between(range(7), storage_con_data, inport_store_cov_data,
                             label='covariance',
                             color='orange',
                             alpha=0.5)
            plt.ylabel('$\lambda_k$')
            plt.xticks(np.arange(0,7),['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
            plt.title('Principle component '+str(i+1))
            if i == 4: # Create legend of figure 4 (lower left)
                plt.legend(loc = 'center', bbox_to_anchor = (1.1,-0.17), ncol = 6,
                           fontsize = 'large', framealpha = 1, columnspacing = 2.5)     
            else:
                assert True, "type_of_contribution not used correct"

    return plt.show(all)


# In[6]:


def MAP(eigen_vectors, eigen_values, data_names, PC_NO, title_plot, filename_plot):
    """
    Parameters
    ----------
    eigen_vectors : maxtrix [N x N]
        Eigen vectors.
    eigen_values : matrix [N x 1]
        Eigen values.
    data_names : List of string [N x 30]
        Name of countries ('alpha-2-code' format).
    PC_NO : scalar integer
        Principal component number (starts from 1).
    title_plot : string
        Title of plot.
    filename_plot : string
        Subplot title.

    Returns
    -------
    Plot
        Map plot.
    """
    VT = pd.DataFrame(data=eigen_vectors, index=data_names)
    variance_explained = []
    for i in eigen_values:
         variance_explained.append((i/sum(eigen_values))*100)
    
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=cartopy.crs.TransverseMercator(20))
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
    ax.coastlines(resolution='10m')
    ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.6,0.8,1), alpha=0.30)
    ax.set_extent ((-9.5, 32, 35, 71), cartopy.crs.PlateCarree())
    ax.gridlines()
    
    europe_not_included = {'AD', 'AL','AX','BY', 'FO', 'GG', 'GI', 'IM', 'IS', 
                           'JE', 'LI', 'MC', 'MD', 'ME', 'MK', 'MT', 'RU', 'SM', 
                           'UA', 'VA', 'XK'}
    shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    name_loop = 'start'
    for country in countries:
        if country.attributes['ISO_A2'] in europe_not_included:
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
        elif country.attributes['REGION_UN'] == 'Europe':
            if country.attributes['NAME'] == 'Norway':
                name_loop = 'NO'
            elif country.attributes['NAME'] == 'France':
                name_loop = 'FR'
            else:
                name_loop = country.attributes['ISO_A2']
            for country_PSA in VT.index.values:
                if country_PSA == name_loop:
                    color_value = VT.loc[country_PSA][PC_NO-1]
                    if color_value <= 0:
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=(1, 0, 0), alpha=(np.min([color_value, 1])), 
                                          linewidth=0.15, edgecolor="black", label=country.attributes['ADM0_A3'])
                    else:
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=(0, 1, 0), alpha=(np.min([color_value, 1])), 
                                          linewidth=0.15, edgecolor="black", label=country.attributes['ADM0_A3'])
        else:
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
    
    plt.title(title_plot)
    plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%'], loc='upper left')
    test = np.zeros([30,30])
    test[0,0]=-1
    test[0,29]=1
    cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0,0),(1,0.333,0.333),(1,0.666,0.666), 'white',(0.666,1,0.666),(0.333,1,0.333),(0,1,0),(0,1,0)])
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    im = ax.imshow(test,cmap=cmap)                
    plt.colorbar(im,cax=cax)
    plt.suptitle(filename_plot,fontsize=20,x=.51,y=0.938)
    
    return (plt.show());


# In[7]:


def BAR(matrix, PC_max, filename, constraints, title, xlabel, suptitle):
    """
    Parameters
    ----------
    matrix : list [len(filename) x 1] of lists of float32 [30 x 1]
        Each sub-list contains the eigen values of each case/scenario
    PC_max : integer
        Number of PC showed on barplot, where the last is summed up i.e. 1, 2, 3, (4-30) which is 4 in total (PC_max=4)
    filename : list of strings
        List of filenames. Used to determine how many bars plotted
    constraints : list of strings
        Used as descriptor for x-axis.
    title : string
        Title of the plot.
    xlabel : string
        label for the x-axis beside the constraint i.e. CO2 or Transmission size.
    suptitle : string (optional)
        Subtitle for plot  (above title). Type 'none' for no subtitle.

    Returns
    -------
    plot
        Returns the final plot.
    """
    fig1 = plt.figure()
    ax = fig1.add_axes([0,0,1,1.5])    
    cmap = plt.get_cmap("Paired")
    j_max = PC_max
    colour_map = cmap(np.arange(j_max))
    lns_fig1 = []
    
    for i in np.arange(0,len(filename)):
        for j in np.arange(0,j_max):
            if j == 0:
                lns_plot = ax.bar(constraints[i], matrix[i][j], color=colour_map[j], edgecolor='black', linewidth=1.2, label=('$k_{' + str(j+1) + '}$'))
            elif (j > 0 and j < (j_max-1)):
                lns_plot = ax.bar(constraints[i], matrix[i][j], bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=('$k_{' + str(j+1) + '}$'))
            else:
                lns_plot = ax.bar(constraints[i], sum(matrix[i][j:29]), bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=('$k_{' + str(j+1) + '}$ - $k_{30}$'))
            if (i==0):
                lns_fig1.append(lns_plot)

    labs = [l.get_label() for l in lns_fig1]
    ax.legend(lns_fig1, labs, bbox_to_anchor = (1,1))
    plt.yticks(range(0, 120, 25))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Data variance of each PC [%]')
    plt.grid(axis='y')
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.68)

    return (plt.show());


# In[8]:


def FFT_plot(T, file_name):
    """
    Parameters
    ----------
    T : Matrix
        Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
    file_name: array of strings
        Name of the datafile there is worked with

    Returns
    -------
    Plot all
    """
    plt.figure(figsize=(18,12))
    plt.subplot(3,2,1)
    freq=np.fft.fftfreq(len(T[0]))  
    FFT=np.fft.fft(T[0])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC1')
    
    plt.subplot(3,2,2)
    freq=np.fft.fftfreq(len(T[1]))  
    FFT=np.fft.fft(T[1])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC2')
    
    plt.subplot(3,2,3)
    freq=np.fft.fftfreq(len(T[2]))  
    FFT=np.fft.fft(T[2])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC3')
    
    plt.subplot(3,2,4)
    freq=np.fft.fftfreq(len(T[3]))  
    FFT=np.fft.fft(T[3])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC4')
    
    plt.subplot(3,2,5)
    freq=np.fft.fftfreq(len(T[4]))  
    FFT=np.fft.fft(T[2])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC5')
    
    plt.subplot(3,2,6)
    freq=np.fft.fftfreq(len(T[5]))  
    FFT=np.fft.fft(T[3])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    plt.plot(1/freq,FFT)
    plt.xscale('log')
    plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    #plt.legend(loc='upper right')
    plt.text(10,0.9,"1/2 Day",ha='right')
    plt.text(22,0.9,"Day",ha='right')
    plt.text(22*7,0.9,"Week",ha='right')
    plt.text(22*7*4,0.9,"Month",ha='right')
    plt.text(22*365,0.9,"Year",ha='right')
    plt.xlabel('Hours')
    plt.title('Fourier Power Spectra for PC6')        
    
    plt.subplots_adjust(wspace=0, hspace=0.28)
    plt.suptitle(file_name,fontsize=20,x=.51,y=0.93) #,x=.51,y=1.07
    
    return plt.show(all)


# ## Map plots

# In[9]:


file = '../data/postnetwork-elec_only_0.125_0.05.h5'
network = pypsa.Network(file)
network.name = file

generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
load = network.loads_t.p_set
mismatch = generation - load

X = mismatch
X_mean = np.mean(X,axis=0)
X_mean = np.array(X_mean.values).reshape(30,1)
X_cent = np.subtract(X,X_mean.T)
c = 1/np.sqrt(np.sum(np.mean(((X_cent.values)**2),axis=0)))
B = c*(X_cent.values)
C_new = np.dot(B.T,B)*1/(8760-1)
C = np.cov(B.T,bias=True) 
eig_val, eig_vec = np.linalg.eig(C) 
T = np.dot(B,eig_vec)


# In[14]:


fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
linewidth = 0.8
panels = ['(a)', '(b)', '(c)', '(d)']
eigen_values = eig_val
eigen_vectors = - eig_vec #invert eigenvector to make the results more intuitive
variance_explained = []
for j in eigen_values:
     variance_explained.append((j/sum(eigen_values))*100)
variance_explained_cumulative = np.cumsum(variance_explained)
data_names = network.loads_t.p.columns
VT = pd.DataFrame(data=eigen_vectors, index=data_names)

for i in range(4):
    ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
    ax[i].coastlines(resolution='110m')
    ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
    ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
    europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                           'ME','MK','MT','RU','SM','UA','VA','XK'}
    shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries_1 = reader.records()
    name_loop = 'start'
    PC_NO = i+1
    for country in countries_1:
        if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
            if country.attributes['NAME'] == 'Norway':
                name_loop = 'NO'
            elif country.attributes['NAME'] == 'France':
                name_loop = 'FR'                
            else:
                name_loop = country.attributes['ISO_A2']
            for country_PSA in VT.index.values:
                if country_PSA == name_loop:
                    color_value = VT.loc[country_PSA][PC_NO-1]
                    if color_value <= 0:
                        color_value = np.absolute(color_value)*1.5
                        ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
                                             alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    else:
                        color_value = np.absolute(color_value)*1.5
                        ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
                                             alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
        else:
            ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
                                 edgecolor="black", label=country.attributes['ADM0_A3'])

    ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
    ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
               fontsize=12, transform=ax[i].transAxes);

cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
plt.subplots_adjust(hspace=0.02, wspace=0.04)
plt.savefig('figures/sec3_combined_PCs.pdf', bbox_inches='tight')


# In[15]:


#file = 'postnetwork-elec_only_0.125_0.05.h5'
#network = pypsa.Network(directory+file)
#network.name = file
data_names = network.loads_t.p.columns
time_index = network.loads_t.p.index
prices = network.buses_t.marginal_price
country_price = prices[data_names] # [€/MWh]
nodal_price = country_price.values
nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)
nodal_price = np.clip(nodal_price, 0, 1000)
eigen_values_prices, eigen_vectors_prices = PCA(nodal_price)[0:2]
eigen_vectors_prices *= -1 #invert eigenvectors to make the results more intuitive


# In[17]:


fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
linewidth = 0.8
panels = ['(a)', '(b)', '(c)', '(d)']
variance_explained_prices = []
for j in eigen_values_prices:
     variance_explained_prices.append((j/sum(eigen_values_prices))*100)
variance_explained_prices_cumulative = np.cumsum(variance_explained_prices)
data_names = network.loads_t.p.columns
VT_prices = pd.DataFrame(data=eigen_vectors_prices, index=data_names)

for i in range(4):
    ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
    ax[i].coastlines(resolution='110m')
    ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
    ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
    europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                           'ME','MK','MT','RU','SM','UA','VA','XK'}
    shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries_1 = reader.records()
    name_loop = 'start'
    PC_NO = i+1
    for country in countries_1:
        if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
            if country.attributes['NAME'] == 'Norway':
                name_loop = 'NO'
            elif country.attributes['NAME'] == 'France':
                name_loop = 'FR'                
            else:
                name_loop = country.attributes['ISO_A2']
            for country_PSA in VT_prices.index.values:
                if country_PSA == name_loop:
                    #print("Match!")
                    color_value = VT_prices.loc[country_PSA][PC_NO-1]
                    #print(color_value)
                    if color_value <= 0:
                        # Color red
                        color_value = np.absolute(color_value)*1.5
                        ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    else:
                        # Color blue
                        color_value = np.absolute(color_value)*1.5
                        ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
        else:  
            ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
                                   edgecolor="black", label=country.attributes['ADM0_A3'])

    ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
    ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained_prices[PC_NO-1],1)) + '%', 
                 fontsize=12, transform=ax[i].transAxes);

cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
plt.subplots_adjust(hspace=0.02, wspace=0.04)
plt.savefig('figures/sec3_combined_PCs_prices.pdf', bbox_inches='tight')


# ## Diurnal and seasonal behaviour

# In[18]:


#%% PCA mismatch
eigen_values_mis, eigen_vectors_mis, variance_explained_mis, norm_const_mis, T_mis = PCA(mismatch)
T_mis *= -1 #invert eigenvectors to make the results more intuitive
T_mis = pd.DataFrame(data=T_mis,index=time_index)

# Season Plot Mismatch
season_plot(T_mis, time_index, 'For the mismatch')


# In[24]:


#%% PCA Electricity nodal price
eigen_values_ENP, eigen_vectors_ENP, variance_explained_ENP, norm_const_ENP, T_ENP = PCA(nodal_price)
T_ENP = pd.DataFrame(data=T_ENP,index=time_index)
T_ENP *= -1 #invert eigenvectors to make the results more intuitive
# Season Plot Prices
season_plot(T_ENP, time_index, 'For the electricity prices')


# ## Fourier power spectra

# In[25]:


# FFT Plot
#FFT_plot(T_mis, 'For the mismatch')


# In[26]:


# FFT Plot
#FFT_plot(T_ENP, 'For the electricity prices')



def pltFFT(T, PC_no, ax):
    freq = np.fft.fftfreq(len(T[PC_no]))  
    FFT = np.fft.fft(T[PC_no])
    #FFT[PC_no] = 0
    FFT = abs(FFT) / max(abs(FFT))
    
    # Only plot half of the FFT spectrum
    freq = freq[:4380]
    FFT = FFT[:4380]
    
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][PC_no]
    ax.plot(1/freq, FFT, marker='.', c=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.vlines(12,0,1, colors="k", linestyles="dotted", linewidth=2) # 1/2 day
    ax.vlines(24,0,1, colors="k", linestyles="dotted", linewidth=2) # day
    ax.vlines(24*7,0,1, colors="k", linestyles="dotted", linewidth=2) # week
    ax.vlines(24*30,0,1, colors="k", linestyles="dotted", linewidth=2) # month
    ax.vlines(24*365,0,1, colors="k", linestyles="dotted", linewidth=2) # year
    ax.text(10, 0.9, "1/2 Day", ha='right')
    ax.text(22, 0.9, "Day", ha='right')
    ax.text(22*7, 0.9, "Week", ha='right')
    ax.text(22*7*4, 0.9, "Month", ha='right')
    ax.text(22*365, 0.9, "Year", ha='right')
    ax.set_xlabel('Hours')
    ax.set_title('Fourier Power Spectra for PC'+str(PC_no+1), fontsize=15.5)

def pltSeasonHourly(T, time_index, ax):
    # Define as dataframe
    T = pd.DataFrame(data=T,index=time_index)
    # Average hour and day
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day
    
    ax.plot(T_avg_hour[0],label='k = 1',marker='.')
    ax.plot(T_avg_hour[1],label='k = 2',marker='.')
    ax.plot(T_avg_hour[2],color="gray",alpha=0.4)
    ax.plot(T_avg_hour[3],color="gray",alpha=0.4)
    ax.plot(T_avg_hour[4],color="gray",alpha=0.4)
    ax.plot(T_avg_hour[5],color="gray",alpha=0.4)
    ax.set_xticks(ticks=range(0,24,2))
    #ax.set_xlim(-0.5,23.5)
    ax.legend(loc='upper left', labelspacing=.1)#,bbox_to_anchor=(1,1))
    ax.set_xlabel("Hour")
    ax.set_ylabel("$a_k$ diurnal")
    ax.set_title("Hourly average over one year", fontsize=15.5)
	
def pltSeasonDaily(T, time_index, eigenvalue, ax, xlabel=True):
    # Define as dataframe
    T = pd.DataFrame(data=T,index=time_index)
    # Average hour and day
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day

    x_ax = range(len(T_avg_day[0])) # X for year plot
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][eigenvalue]
    ax.plot(x_ax,T_avg_day[eigenvalue],label='$\lambda_1$',c=color)
    #ax.plot(x_ax,T_avg_day[1],label='$\lambda_2$')
    #ax.plot(x_ax,T_avg_day[2],label='$\lambda_3$')
    #ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    #ax.set_xlim(-4,368)
    if xlabel:
        ax.set_xlabel("Day")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("$a_k$ seasonal")
    ax.set_title("Daily average for $k = "+str(eigenvalue+1)+"$", fontsize=15.5)

#----------------------------------#

filename = ["postnetwork-elec_only_0_0.05.h5",
            "postnetwork-elec_only_0.0625_0.05.h5",
            "postnetwork-elec_only_0.125_0.05.h5",
            "postnetwork-elec_only_0.25_0.05.h5",
            "postnetwork-elec_only_0.375_0.05.h5"]
dic = '../data/'

# Variable for principal components for plotting later
PC1_con2 = np.zeros((5,11))
PC2_con2 = np.zeros((5,11))
PC3_con2 = np.zeros((5,11))
PC4_con2 = np.zeros((5,11))
PC5_con2 = np.zeros((5,11))
PC6_con2 = np.zeros((5,11))
PC_con2 = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    # User info:
    print("\nCalculating for: ",filename[i])
    # Create network from previous file
    network = pypsa.Network(dic+filename[i])

    #%% Define index and columns
    # Index
    time_index = network.loads_t.p_set.index 
    # Columns
    country_column = network.loads_t.p_set.columns
    
    #%% Calculating mismatch
    # Defining dispatched electricity generation
    generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
    # Defining load 
    load = network.loads_t.p_set
    # Calculate mismatch
    mismatch = generation - load # Using available electricity generation
    # Collecting mismatch terms
    gen_grouped = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum().values
    mismatch_terms = pd.DataFrame({'wind': gen_grouped[:,0]+gen_grouped[:,1],
                                    'ror': gen_grouped[:,2],
                                    'solar': gen_grouped[:,3],
                                    'load': load.sum(axis=1).values},index=time_index)
    
    #%% Collecting technologies per country
    # Combine the load at every timestep for all countries
    load_EU = np.sum(load, axis=1)
    # Dataframe (array) for different generator technologies
    generator_wind = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)
    generator_solar = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)
    generator_hydro = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)
    
    # Counter for positioning in generator data
    counter = 0
    for j in network.generators.index:
        # Current value to insert into correct array and position
        value = np.array(network.generators_t.p)[:,counter]
        # Check for wind, solar and hydro
        if (j[-4:] == "wind"):
            generator_wind[j[0:2]] = generator_wind[j[0:2]] + value
        elif (j[-5:] == "solar"):
            generator_solar[j[0:2]] = generator_solar[j[0:2]] + value
        elif (j[-3:] == "ror"):
            generator_hydro[j[0:2]] = generator_hydro[j[0:2]] + value
        # Increase value of counter by 1
        counter +=1
    
    # Mean values
    wind_mean = np.mean(generator_wind,axis=0)
    solar_mean = np.mean(generator_solar,axis=0)
    hydro_mean = np.mean(generator_hydro,axis=0)
    load_mean = np.mean(load,axis=0)
    
    # Centering data
    wind_cent = np.subtract(generator_wind,wind_mean.T)
    solar_cent = np.subtract(generator_solar,solar_mean.T)
    hydro_cent = np.subtract(generator_hydro,hydro_mean.T)
    load_cent = np.subtract(load,load_mean.T)
    
    #check = wind_cent+solar_cent+hydro_cent-load_cent.values
    
    #%% Principal Component Analysis
    
    # Defining data
    X = mismatch
    
    # Mean of data
    X_mean = np.mean(X,axis=0) # axis=0, mean at each colume 
    X_mean = np.array(X_mean.values).reshape(30,1) # Define as an array
    
    # Calculate centered data
    X_cent = np.subtract(X,X_mean.T)
    
    # Calculate normalization constant
    c = 1/np.sqrt(np.sum(np.mean(((X_cent.values)**2),axis=0)))
    
    # Normalize the centered data
    B = c*(X_cent.values)
    
    # Convariance of normalized and centered data
    C_new = np.dot(B.T,B)*1/(8760-1)
    C = np.cov(B.T,bias=True) 
    
    # Calculate eigen values and eigen vectors 
    assert np.size(C) <= 900, "C is too big" # Checks convariance size, if to large then python will be stuck on the eigen problem
    eig_val, eig_vec = np.linalg.eig(C) # using numpy's function as it scales eigen values
    
    # Calculate amplitude
    T = np.dot(B,eig_vec)
    
    #%% eigen values contribution
    
    # Component contribution
    # Wind
    wind_con2 = np.dot(wind_cent,eig_vec)
    # Solar
    solar_con2 = np.dot(solar_cent,eig_vec)
    # Hydro
    ror_con2 = np.dot(hydro_cent,eig_vec)
    # Load
    load_con2 = np.dot(load_cent,eig_vec)
    # Sum (with -L) of this is equal to T
    
    # Eigenvalues contribution
    # Wind
    lambda_W = (c**2)*(np.mean((wind_con2**2),axis=0))
    # Solar
    lambda_S = (c**2)*(np.mean((solar_con2**2),axis=0))
    # Hydro
    lambda_H = (c**2)*(np.mean((ror_con2**2),axis=0))
    # Load
    lambda_L = (c**2)*(np.mean((load_con2**2),axis=0))
    # Wind+Solar
    lambda_WS = (c**2)*2*(np.mean((wind_con2*solar_con2),axis=0))
    # Wind+Hydro
    lambda_WH = (c**2)*2*(np.mean((wind_con2*ror_con2),axis=0))
    # Hydro+Solar
    lambda_HS = (c**2)*2*(np.mean((ror_con2*solar_con2),axis=0))
    # Wind+Load
    lambda_WL = (c**2)*2*(np.mean((wind_con2*load_con2),axis=0))
    # Load+Solar
    lambda_LS = (c**2)*2*(np.mean((load_con2*solar_con2),axis=0))
    # Load+Hydro
    lambda_LH = (c**2)*2*(np.mean((load_con2*ror_con2),axis=0))
    
    # Collecting terms
    lambda_collect_wmin = pd.DataFrame({'wind':             lambda_W,
                                       'solar':             lambda_S,
                                       'RoR':               lambda_H,
                                       'load':              lambda_L,
                                       'wind/\nsolar':      lambda_WS,
                                       'wind/\nRoR':        lambda_WH,
                                       'RoR/\nsolar':       lambda_HS,
                                       'wind/\nload':      -lambda_WL,
                                       'load/\nsolar':     -lambda_LS,
                                       'load/\nRoR':       -lambda_LH,
                                       })
    lambda_collect_nmin = pd.DataFrame({'wind':             lambda_W,
                                       'solar':             lambda_S,
                                       'RoR':               lambda_H,
                                       'load':              lambda_L,
                                       'wind/\nsolar':      lambda_WS,
                                       'wind/\nRoR':        lambda_WH,
                                       'RoR/\nsolar':       lambda_HS,
                                       'wind/\nload':       lambda_WL,
                                       'load/\nsolar':      lambda_LS,
                                       'load/\nRoR':        lambda_LH,
                                       })
    
    lambda_tot = sum([+lambda_W,
                     +lambda_S,
                     +lambda_H,
                     +lambda_L,
                     +lambda_WS,
                     +lambda_WH,
                     +lambda_HS,
                     -lambda_WL,
                     -lambda_LS,
                     -lambda_LH
                     ])
    
    lambda_collect_all = pd.DataFrame({'wind':              lambda_W,
                                       'solar':             lambda_S,
                                       'RoR':               lambda_H,
                                       'load':              lambda_L,
                                       'wind/\nsolar':      lambda_WS,
                                       'wind/\nRoR':        lambda_WH,
                                       'RoR/\nsolar':       lambda_HS,
                                       'wind/\nload':      -lambda_WL,
                                       'load/\nsolar':     -lambda_LS,
                                       'load/\nRoR':       -lambda_LH,
                                       'total':             lambda_tot
                                       })
    
    if i == 2:
        #%% Plotting of eigen values contribution
        plt.figure(figsize=[14,16])
        for nn in range(6):
            lambda_collect_procent = lambda_collect_wmin[nn:nn+1]/lambda_tot[nn]*100 # percentage
            #lambda_collect_procent = lambda_collect_wmin[nn:nn+1]/lambda_tot[nn]*100 # relative
            plt.subplot(3,2,nn+1)
            plt.bar(lambda_collect_wmin.columns,lambda_collect_procent.values[0])
            plt.title('PC'+str(nn+1)+': '+str(round(lambda_tot[nn],3)))
            plt.ylabel('Influance [%]')
            plt.ylim([-50,125])
            plt.grid(axis='y',alpha=0.5)
            for k in range(10):
                if lambda_collect_procent.values[:,k] < 0:
                    v = lambda_collect_procent.values[:,k] - 6.5
                else:
                    v = lambda_collect_procent.values[:,k] + 2.5
                plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
        plt.suptitle(filename[i],fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07     
        plt.show(all)

    #%% Save data for PC1-PC6
    PC1_con2[i] = lambda_collect_all[0:1].values
    PC2_con2[i] = lambda_collect_all[1:2].values
    PC3_con2[i] = lambda_collect_all[2:3].values
    PC4_con2[i] = lambda_collect_all[3:4].values
    PC5_con2[i] = lambda_collect_all[4:5].values
    PC6_con2[i] = lambda_collect_all[5:6].values

#%% Data handling
PC_con2.append(PC1_con2)
PC_con2.append(PC2_con2)
PC_con2.append(PC3_con2)
PC_con2.append(PC4_con2)
PC_con2.append(PC5_con2)
PC_con2.append(PC6_con2)

#----------------------------------#

filename = ["postnetwork-elec_only_0_0.05.h5",
            "postnetwork-elec_only_0.0625_0.05.h5",
            "postnetwork-elec_only_0.125_0.05.h5",
            "postnetwork-elec_only_0.25_0.05.h5",
            "postnetwork-elec_only_0.375_0.05.h5"]
dic = '../data/' # Location of files

# Variable for principal components for plotting later
PC1_con1 = np.zeros((5,11))
PC2_con1 = np.zeros((5,11))
PC3_con1 = np.zeros((5,11))
PC4_con1 = np.zeros((5,11))
PC5_con1 = np.zeros((5,11))
PC6_con1 = np.zeros((5,11))
PC_con1_new = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    network = pypsa.Network(dic+filename[i])
    data_names = network.loads_t.p.columns
    
    #%% Define index and columns
    time_index = network.loads_t.p_set.index 
    country_column = network.loads_t.p_set.columns
    
    #%% Calculating mismatch
    # Defining dispatched electricity generation
    generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
    # Defining load 
    load = network.loads_t.p_set
    # Calculate mismatch
    mismatch = generation - load # Using available electricity generation
    
    #%% FIX FOR 0x CURRENT TRANSMISSION LINKS IS UNSORTED
    links_unsorted = network.links_t.p0
    if i == 0:
        sorted_list = ['AT OCGT', 'BA OCGT', 'BE OCGT', 'BG OCGT', 'CH OCGT', 'CZ OCGT',
               'DE OCGT', 'DK OCGT', 'EE OCGT', 'ES OCGT', 'FI OCGT', 'FR OCGT',
               'GB OCGT', 'GR OCGT', 'HR OCGT', 'HU OCGT', 'IE OCGT', 'IT OCGT',
               'LT OCGT', 'LU OCGT', 'LV OCGT', 'NL OCGT', 'NO OCGT', 'PL OCGT',
               'PT OCGT', 'RO OCGT', 'RS OCGT', 'SE OCGT', 'SI OCGT', 'SK OCGT',
               'AT H2 Electrolysis', 'BA H2 Electrolysis', 'BE H2 Electrolysis',
               'BG H2 Electrolysis', 'CH H2 Electrolysis', 'CZ H2 Electrolysis',
               'DE H2 Electrolysis', 'DK H2 Electrolysis', 'EE H2 Electrolysis',
               'ES H2 Electrolysis', 'FI H2 Electrolysis', 'FR H2 Electrolysis',
               'GB H2 Electrolysis', 'GR H2 Electrolysis', 'HR H2 Electrolysis',
               'HU H2 Electrolysis', 'IE H2 Electrolysis', 'IT H2 Electrolysis',
               'LT H2 Electrolysis', 'LU H2 Electrolysis', 'LV H2 Electrolysis',
               'NL H2 Electrolysis', 'NO H2 Electrolysis', 'PL H2 Electrolysis',
               'PT H2 Electrolysis', 'RO H2 Electrolysis', 'RS H2 Electrolysis',
               'SE H2 Electrolysis', 'SI H2 Electrolysis', 'SK H2 Electrolysis',
               'AT H2 Fuel Cell', 'BA H2 Fuel Cell', 'BE H2 Fuel Cell',
               'BG H2 Fuel Cell', 'CH H2 Fuel Cell', 'CZ H2 Fuel Cell',
               'DE H2 Fuel Cell', 'DK H2 Fuel Cell', 'EE H2 Fuel Cell',
               'ES H2 Fuel Cell', 'FI H2 Fuel Cell', 'FR H2 Fuel Cell',
               'GB H2 Fuel Cell', 'GR H2 Fuel Cell', 'HR H2 Fuel Cell',
               'HU H2 Fuel Cell', 'IE H2 Fuel Cell', 'IT H2 Fuel Cell',
               'LT H2 Fuel Cell', 'LU H2 Fuel Cell', 'LV H2 Fuel Cell',
               'NL H2 Fuel Cell', 'NO H2 Fuel Cell', 'PL H2 Fuel Cell',
               'PT H2 Fuel Cell', 'RO H2 Fuel Cell', 'RS H2 Fuel Cell',
               'SE H2 Fuel Cell', 'SI H2 Fuel Cell', 'SK H2 Fuel Cell',
               'AT battery charger', 'BA battery charger', 'BE battery charger',
               'BG battery charger', 'CH battery charger', 'CZ battery charger',
               'DE battery charger', 'DK battery charger', 'EE battery charger',
               'ES battery charger', 'FI battery charger', 'FR battery charger',
               'GB battery charger', 'GR battery charger', 'HR battery charger',
               'HU battery charger', 'IE battery charger', 'IT battery charger',
               'LT battery charger', 'LU battery charger', 'LV battery charger',
               'NL battery charger', 'NO battery charger', 'PL battery charger',
               'PT battery charger', 'RO battery charger', 'RS battery charger',
               'SE battery charger', 'SI battery charger', 'SK battery charger',
               'AT battery discharger', 'BA battery discharger',
               'BE battery discharger', 'BG battery discharger',
               'CH battery discharger', 'CZ battery discharger',
               'DE battery discharger', 'DK battery discharger',
               'EE battery discharger', 'ES battery discharger',
               'FI battery discharger', 'FR battery discharger',
               'GB battery discharger', 'GR battery discharger',
               'HR battery discharger', 'HU battery discharger',
               'IE battery discharger', 'IT battery discharger',
               'LT battery discharger', 'LU battery discharger',
               'LV battery discharger', 'NL battery discharger',
               'NO battery discharger', 'PL battery discharger',
               'PT battery discharger', 'RO battery discharger',
               'RS battery discharger', 'SE battery discharger',
               'SI battery discharger', 'SK battery discharger', 'AT-CH', 'AT-CZ',
               'AT-DE', 'AT-HU', 'AT-IT', 'AT-SI', 'BA-HR', 'BA-RS', 'BG-GR',
               'BG-RO', 'BG-RS', 'CH-DE', 'CH-IT', 'CZ-DE', 'CZ-SK', 'DE-DK',
               'DE-LU', 'DE-SE', 'EE-LV', 'FI-EE', 'FI-SE', 'FR-BE', 'FR-CH',
               'FR-DE', 'FR-ES', 'FR-GB', 'FR-IT', 'GB-IE', 'GR-IT', 'HR-HU',
               'HR-RS', 'HR-SI', 'HU-RS', 'HU-SK', 'IT-SI', 'LV-LT', 'NL-BE',
               'NL-DE', 'NL-GB', 'NL-NO', 'NO-DK', 'NO-SE', 'PL-CZ', 'PL-DE',
               'PL-LT', 'PL-SE', 'PL-SK', 'PT-ES', 'RO-HU', 'RO-RS', 'SE-DK',
               'SE-LT']
        links_sorted = pd.DataFrame(data=0, index=links_unsorted.index, columns=sorted_list)
        for k in np.arange(0,len(sorted_list)):
            link = links_sorted.columns[k]
            links_sorted[link] = links_unsorted[link]
        # Updated with sorted list
        network.links_t.p0 = links_sorted
    
    #%% Backup generator
    # Efficiency (can be seen at (network.links.efficiency))
    eff_gas = network.links.efficiency['DK OCGT'] #0.39
    # output gas backup generator
    store_gas_out = pd.DataFrame(np.array(network.links_t.p0)[:,0:30], columns=(data_names))
    # With efficiency
    store_gas_out_eff = store_gas_out * eff_gas
    
    #%% Storage
    # Efficiency for different storages (can be seen at (network.links.efficiency))
    eff_H2_charge = network.links.efficiency['DK H2 Electrolysis'] #0.8
    eff_H2_discharge = network.links.efficiency['DK H2 Fuel Cell'] #0.58
    eff_battery_charge = network.links.efficiency['DK battery charger'] #0.9
    eff_battery_discharge = network.links.efficiency['DK battery discharger'] #0.9
    eff_PHS = network.storage_units.efficiency_dispatch['AT PHS'] #0.866
    # H2 storage
    store_H2_in = pd.DataFrame(np.array(network.links_t.p0)[:,30:60], columns=(data_names))
    store_H2_out = pd.DataFrame(np.array(network.links_t.p0)[:,60:90], columns=(data_names))
    # Battery stoage
    store_batttery_in = pd.DataFrame(np.array(network.links_t.p0)[:,90:120], columns=(data_names))
    store_battery_out = pd.DataFrame(np.array(network.links_t.p0)[:,120:150], columns=(data_names))
    # PHS and hydro
    PHS_and_hydro = network.storage_units_t.p.groupby([network.storage_units.carrier, network.storage_units.bus],axis=1).sum()
    PHS_and_hydro_val = PHS_and_hydro.values
    PHS = PHS_and_hydro["PHS"]
    hydro = PHS_and_hydro["hydro"]
    store_PHS = np.zeros((8760,30)) # All countries
    store_hydro = np.zeros((8760,30)) # All countries
    for k in range(len(data_names)):
        if data_names[k] in PHS.columns: # Checks if the country is in PHS
            store_PHS[:,k] += PHS[data_names[k]] # Adds the value
    for k in range(len(data_names)):
        if data_names[k] in hydro.columns: # Checks if the country is in PHS
            store_hydro[:,k] += hydro[data_names[k]] # Adds the value
    store_PHS = pd.DataFrame(store_PHS, columns=data_names)
    store_hydro = pd.DataFrame(store_hydro, columns=data_names)
    # With efficiency
    store_H2_in_eff = store_H2_in #* 1/eff_H2_charge
    store_H2_out_eff = store_H2_out * eff_H2_discharge
    store_battery_in_eff = store_batttery_in #* 1/eff_battery_charge
    store_battery_out_eff = store_battery_out * eff_battery_discharge
    store_PHS_eff = store_PHS #* eff_PHS
    store_hydro_eff = store_hydro 
    # Sum of all storage
    store_sum = - store_H2_in + store_H2_out - store_batttery_in + store_battery_out + store_PHS
    # Sum of all storage including efficicency
    store_sum_eff = - store_H2_in_eff + store_H2_out_eff - store_battery_in_eff + store_battery_out_eff + store_PHS_eff
    
    #%% Import/export (links)
    # Country links (p0/p1 = active power at bus0/bus1)
    # Values of p0 links:   network.links_t.p0
    # Values of p0 links:   network.links_t.p1
    # Names of all link:    network.links_t.p0.columns
    country_link_names = np.array(network.links_t.p0.columns)[150:]
    # Link p0 is the same as negative (-) p1
    # Turn into array
    country_links_p0 = np.array(network.links_t.p0)[:,150:]
    # Turn into dataframe
    country_links_p0 = pd.DataFrame(country_links_p0, columns=(country_link_names))
    # Turn into array
    country_links_p1 = np.array(network.links_t.p1)[:,150:]
    # Turn into dataframe
    country_links_p1 = pd.DataFrame(country_links_p1, columns=(country_link_names))
    # Sum all bus0 exports values
    sum_bus0 = (country_links_p0.groupby(network.links.bus0,axis=1).sum())
    # Sum all bus1 imports values and minus
    sum_bus1 = (country_links_p1.groupby(network.links.bus1,axis=1).sum())
    
    # Creating empty matrix
    country_links = np.zeros((8760,30))
    # loop that adds bus0 and bus1 together (remember bus1 is minus value)
    for k in range(len(data_names)):   
        if data_names[k] in sum_bus0.columns: # Checks if the country is in bus0
            country_links[:,k] += sum_bus0[data_names[k]] # Adds the value for the country to the collected link function
        if data_names[k] in sum_bus1.columns: # Checks if the country is in bus1
            country_links[:,k] += sum_bus1[data_names[k]] # Adds the value for the country to the collected link function
    # Define the data as a pandas dataframe with country names and timestampes
    country_links = pd.DataFrame(data=country_links, index=network.loads_t.p.index, columns=data_names)
    
    #%% Principal Component Analysis
    # Defining data
    X = mismatch
    # Mean of data
    X_mean = np.mean(X,axis=0) # axis=0, mean at each colume 
    X_mean = np.array(X_mean.values).reshape(30,1) # Define as an array
    # Calculate centered data
    X_cent = np.subtract(X,X_mean.T)
    # Calculate normalization constant
    c = 1/np.sqrt(np.sum(np.mean(((X_cent.values)**2),axis=0)))
    # Normalize the centered data
    B = c*(X_cent.values)
    # Convariance of normalized and centered data
    C_new = np.dot(B.T,B)*1/(8760-1)
    C = np.cov(B.T,bias=True) 
    # Calculate eigen values and eigen vectors 
    assert np.size(C) <= 900, "C is too big" # Checks convariance size, if to large then python will be stuck on the eigen problem
    eig_val, eig_vec = np.linalg.eig(C) # using numpy's function as it scales eigen values
    # Calculate amplitude
    T = np.dot(B,eig_vec)
    
    #%% CHECKER
    checker = country_links - store_sum_eff.values - store_gas_out_eff.values - store_hydro_eff.values
    checker2 = checker - X.values
    
    #%% Centering responce data
    # Mean values
    backup_mean = np.mean(store_gas_out_eff,axis=0)
    inport_export_mean = np.mean(country_links,axis=0)
    storage_mean = np.mean(store_sum_eff,axis=0)
    hydro_reservoir_mean = np.mean(store_hydro_eff,axis=0)
    
    # Centering data
    backup_cent = - np.subtract(store_gas_out_eff,backup_mean.T) # MINUS ADDED?
    inport_export_cent = np.subtract(country_links,inport_export_mean.T)
    storage_cent = - np.subtract(store_sum_eff,storage_mean.T) # MINUS ADDED?
    hydro_reservoir_cent = - np.subtract(store_hydro_eff,hydro_reservoir_mean.T)
    
    #check = backup_cent + inport_export_cent.values + storage_cent.values
    
    #%% eigen values contribution
    
    # Component contribution
    # Backup generator
    backup_con1 = np.dot(backup_cent,eig_vec)
    # inport/export
    inport_export_con1 = np.dot(inport_export_cent,eig_vec)
    # storage technologies
    storage_con1 = np.dot(storage_cent,eig_vec)
    # hydro reservoir
    hydro_reservoir_con1 = np.dot(hydro_reservoir_cent,eig_vec)
    
    # Sum (with -L) of this is equal to T
    #check = (backup_con1 + inport_export_con1 + storage_con1)*c
    
    # Eigenvalues contribution
    # Backup
    lambda_B = (c**2)*(np.mean((backup_con1**2),axis=0))
    # inport/export
    lambda_P = (c**2)*(np.mean((inport_export_con1**2),axis=0))
    # storage technologies
    lambda_DeltaS = (c**2)*(np.mean((storage_con1**2),axis=0))
    # hydro reservoir
    lambda_H = (c**2)*(np.mean((hydro_reservoir_con1**2),axis=0))
    # Backup + inport/export
    lambda_BP = (c**2)*2*(np.mean((backup_con1*inport_export_con1),axis=0))
    # Backup + storage technologies
    lambda_BdeltaS = (c**2)*2*(np.mean((backup_con1*storage_con1),axis=0))
    # inport/export + storage technologies
    lambda_PdeltaS = (c**2)*2*(np.mean((inport_export_con1*storage_con1),axis=0))
    # B + H
    lambda_BH = (c**2)*2*(np.mean((backup_con1*hydro_reservoir_con1),axis=0))
    # S + H
    lambda_deltaSH = (c**2)*2*(np.mean((storage_con1*hydro_reservoir_con1),axis=0))
    # P + H
    lambda_PH = (c**2)*2*(np.mean((inport_export_con1*hydro_reservoir_con1),axis=0))
    
    # Collecting terms
    lambda_collect = pd.DataFrame({'backup':                    lambda_B,
                                   'import &\nexport':          lambda_P,
                                   'storage':                   lambda_DeltaS,
                                   'hydro reservoir':           lambda_H,
                                   'backup/\ninport/export':    lambda_BP,
                                   'backup/\nstorage':          lambda_BdeltaS,
                                   'backup/\nhydro reservoir':  lambda_BH,
                                   'inport/export/\nstorage':   lambda_PdeltaS,
                                   'inport/export/\nhydro reservoir':   lambda_PH,
                                   'storage/\nhydro reservoir': lambda_deltaSH})
    lambda_tot = sum([+lambda_B,
                      +lambda_P,
                      +lambda_DeltaS,
                      +lambda_H,
                      +lambda_BP,
                      +lambda_BdeltaS,
                      +lambda_BH,
                      +lambda_PdeltaS,
                      +lambda_PH,
                      +lambda_deltaSH])
    lambda_collect_all = pd.DataFrame({'backup':                    lambda_B,
                                       'import &\nexport':          lambda_P,
                                       'storage':                   lambda_DeltaS,
                                       'hydro reservoir':           lambda_H,
                                       'backup/\ninport/export':    lambda_BP,
                                       'backup/\nstorage':          lambda_BdeltaS,
                                       'backup/\nhydro reservoir':  lambda_BH,
                                       'inport/export/\nstorage':   lambda_PdeltaS,
                                       'inport/export/\nhydro reservoir':   lambda_PH,
                                       'storage/\nhydro reservoir': lambda_deltaSH,
                                       'total':                     lambda_tot})
    
    #%% Save data for PC1-PC6
    PC1_con1[i] = lambda_collect_all[0:1].values
    PC2_con1[i] = lambda_collect_all[1:2].values
    PC3_con1[i] = lambda_collect_all[2:3].values
    PC4_con1[i] = lambda_collect_all[3:4].values
    PC5_con1[i] = lambda_collect_all[4:5].values
    PC6_con1[i] = lambda_collect_all[5:6].values   
    
#%% Data handling
PC_con1_new.append(PC1_con1)
PC_con1_new.append(PC2_con1)
PC_con1_new.append(PC3_con1)
PC_con1_new.append(PC4_con1)
PC_con1_new.append(PC5_con1)
PC_con1_new.append(PC6_con1)

#----------------------------------#



##### Plot mismatch (Fig 5) #####

fig = plt.figure(constrained_layout=True, figsize=(16,10))
gs = fig.add_gridspec(4, 2)
# ax1 = fig.add_subplot(gs[0:2, 0])
# pltFFT(T_mis, 0, ax1)
# ax1.text(-0.08, 1.03, '(a)', fontsize=17, transform=ax1.transAxes);

# ax2 = fig.add_subplot(gs[0:2, 1])
# pltFFT(T_mis, 1, ax2)
# ax2.text(-0.08, 1.03, '(b)', fontsize=17, transform=ax2.transAxes);

ax3 = fig.add_subplot(gs[0:2, 0])
pltSeasonHourly(T_mis, time_index, ax3)  #invert eigenvector to make the results more intuitive
ax3.text(-0.08, 1.03, '(a)', fontsize=17, transform=ax3.transAxes);

ax4 = fig.add_subplot(gs[0, 1])
pltSeasonDaily(T_mis, time_index, 0, ax4, xlabel=False)
ax4.set_ylim(-0.85, 0.85)
ax4.grid(True, which='major', axis='y')  #invert eigenvector to make the results more intuitive
ax4.text(-0.08, 1.03, '(b)', fontsize=17, transform=ax4.transAxes);

ax5 = fig.add_subplot(gs[1, 1])
pltSeasonDaily(T_mis, time_index, 1, ax5)
ax5.set_ylim(-0.85, 0.85); 
ax5.grid(True, which='major', axis='y')  #invert eigenvector to make the results more intuitive
ax5.text(-0.08, 1.03, '(c)', fontsize=17, transform=ax5.transAxes);



# Plotting of eigen values contribution
ax6 = fig.add_subplot(gs[2:4, 0])
fs=13
total_pc1 = PC_con2[0][2][-1]
total_pc2 = PC_con2[1][2][-1]
data_pc1 = pd.Series(data=(PC_con2[0][2]/total_pc1)[:-1]*100,  # divided by total and in percent
                     index=['Wind','Solar','RoR','Load','Wind/Solar','Wind/RoR',
                            'RoR/Solar','Wind/Load','Load/Solar','Load/RoR'])
data_pc2 = pd.Series(data=(PC_con2[1][2]/total_pc2)[:-1]*100,  # divided by total and in percent
                     index=['Wind','Solar','RoR','Load','Wind/Solar','Wind/RoR',
                            'RoR/Solar','Wind/Load','Load/Solar','Load/RoR'])

ax6.bar(data_pc1.index, data_pc1.values, align='edge', width=-0.4, label='PC 1 ('+str(round(total_pc1*100,1))+'%)')
ax6.bar(data_pc2.index, data_pc2.values, align='edge', width=0.4, label='PC 2 ('+str(round(total_pc2*100,1))+'%)')

ax6.set_title('Generator contribution', fontdict={'fontsize':15.5})
ax6.set_ylabel('Influence [%]', fontdict={'fontsize':fs+1})
ax6.set_xticklabels(data_pc1.index, rotation=34, ha='right', rotation_mode='anchor', fontdict={'fontsize':fs+.5})
ax6.yaxis.set_tick_params(labelsize=fs+.5)
ax6.set_ylim([-50, 115])
ax6.grid(axis='y', alpha=0.5)

for k in range(len(data_pc1)):
    if data_pc1.values[k] > 0 and data_pc1.values[k] < 100:
        ax6.text(x=k-0.2,y=data_pc1.values[k]+2, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='bottom', size=11.5, rotation=90)
    elif data_pc1.values[k] > 100:
        ax6.text(x=k-0.2,y=data_pc1.values[k]-1, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='top', size=11.5, rotation=90, color='white')
    elif data_pc1.values[k] < -20:
        ax6.text(x=k-0.2,y=data_pc1.values[k]+3, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='bottom', size=11.5, rotation=90, color='white')
    else:
        ax6.text(x=k-0.2,y=data_pc1.values[k]-2, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='top', size=11.5, rotation=90)
    if data_pc2.values[k] > 0:
        ax6.text(x=k+0.2,y=data_pc2.values[k]+2, s=str(round(float(data_pc2.values[k]),1))+'%', ha='center', va='bottom', size=11.5, rotation=90)
    else:
        ax6.text(x=k+0.2,y=data_pc2.values[k]-2, s=str(round(float(data_pc2.values[k]),1))+'%', ha='center', va='top', size=11.5, rotation=90)
ax6.legend(loc='upper right', ncol=1, fontsize=fs)
ax6.text(-0.08, 1.03, '(d)', fontsize=17, transform=ax6.transAxes);


ax7 = fig.add_subplot(gs[2:4, 1])

total_pc1 = PC_con1_new[0][2][-1]
total_pc2 = PC_con1_new[1][2][-1]
data_pc1 = pd.Series(data=(PC_con1_new[0][2]/total_pc1)[:-1]*100,  # divided by total and in percent
                     index=['Backup','Import & export','Storage','Hydro reservoir','Import & export/\nBackup','Backup/Storage','Backup/\nHydro reservoir',
                            'Import & export/\nStorage','Import & export/\nHydro reservoir','Storage/\nHydro reservoir'])
data_pc2 = pd.Series(data=(PC_con1_new[1][2]/total_pc2)[:-1]*100,  # divided by total and in percent
                     index=['Backup','Import & export','Storage','Hydro reservoir','Import & export/\nBackup','Backup/Storage','Backup/\nHydro reservoir',
                            'Import & export/\nStorage','Import & export/\nHydro reservoir','Storage/\nHydro reservoir'])

ax7.bar(data_pc1.index, data_pc1.values, align='edge', width=-0.4, label='PC 1 ('+str(round(total_pc1*100,1))+'%)')
ax7.bar(data_pc2.index, data_pc2.values, align='edge', width=0.4, label='PC 2 ('+str(round(total_pc2*100,1))+'%)')

ax7.set_title('Response contribution', fontdict={'fontsize':15.5})
ax7.set_ylabel('Influence [%]', fontdict={'fontsize':fs+1})
ax7.set_xticklabels(data_pc1.index, rotation=34, ha='right', rotation_mode='anchor', fontdict={'fontsize':fs+.5, 'linespacing':0.75})

ax7.yaxis.set_tick_params(labelsize=fs+.5)
ax7.set_ylim([-50, 115])
ax7.grid(axis='y', alpha=0.5)

for k in range(len(data_pc1)):
    if data_pc1.values[k] > 0 and data_pc1.values[k] < 100 or data_pc1.values[k] < -20:
        ax7.text(x=k-0.2,y=data_pc1.values[k]+2, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='bottom', size=11.5, rotation=90)
    else:
        ax7.text(x=k-0.2,y=data_pc1.values[k]-2, s=str(round(float(data_pc1.values[k]),1))+'%', ha='center', va='top', size=11.5, rotation=90)
    if data_pc2.values[k] > 0:
        ax7.text(x=k+0.2,y=data_pc2.values[k]+2, s=str(round(float(data_pc2.values[k]),1))+'%', ha='center', va='bottom', size=11.5, rotation=90)
    else:
        ax7.text(x=k+0.2,y=data_pc2.values[k]-2, s=str(round(float(data_pc2.values[k]),1))+'%', ha='center', va='top', size=11.5, rotation=90)
ax7.legend(loc='upper right', ncol=1, fontsize=fs);
ax7.text(-0.08, 1.03, '(e)', fontsize=17, transform=ax7.transAxes);

# Save figure: Responses as a Function of Transmission Link Sizes
fig.savefig('figures/sec3_1_combined_mismatch_fft_averages.pdf', transparent=False, dpi=300, bbox_inches='tight')

#%%

##### Plot prices (Fig 10) #####

fig = plt.figure(constrained_layout=True, figsize=(16,4))
gs = fig.add_gridspec(2, 2)
# ax1 = fig.add_subplot(gs[0:2, 0])
# pltFFT(T_ENP, 0, ax1)
# ax1.text(-0.08, 1.03, '(a)', fontsize=17, transform=ax1.transAxes);

# ax2 = fig.add_subplot(gs[0:2, 1])
# pltFFT(T_ENP, 1, ax2)
# ax2.text(-0.08, 1.03, '(b)', fontsize=17, transform=ax2.transAxes);

ax3 = fig.add_subplot(gs[0:2, 0])
pltSeasonHourly(T_ENP, time_index, ax3)
ax3.text(-0.08, 1.03, '(a)', fontsize=17, transform=ax3.transAxes);

ax4 = fig.add_subplot(gs[0, 1])
pltSeasonDaily(T_ENP, time_index, 0, ax4, xlabel=False)
ax4.grid(True, which='major', axis='y')
ax4.text(-0.08, 1.03, '(b)', fontsize=17, transform=ax4.transAxes);

ax5 = fig.add_subplot(gs[1, 1])
pltSeasonDaily(T_ENP, time_index, 1, ax5)
ax5.grid(True, which='major', axis='y')
ax5.text(-0.08, 1.03, '(c)', fontsize=17, transform=ax5.transAxes);

fig.savefig('figures/sec3_2_combined_prices_fft_averages.pdf', transparent=False, dpi=300, bbox_inches='tight')