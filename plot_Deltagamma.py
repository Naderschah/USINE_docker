#from ROOT import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
plt.rcParams["figure.figsize"] = (10,12)

#fig, ax = plt.subplots(1, 1, figsize= (10,14))

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def handle_input_file(df):
    # Input file handling
    df = df.drop(0)
    df = df.drop(1)
    df = df.drop(2)
    df = df.drop(3)
    df = df.drop(4)

    df['new'] = df['# USINE V4.0'].str.split(' ')
    df_temp = df['new']

    df_final = pd.DataFrame(columns=['Rig', 'Flux'])

    list_temp = df_temp.tolist()
    for i in range(0, len(list_temp)):
        list_ = list(filter(None, list_temp[i]))
        df_final.loc[i] = list_

    return df_final.astype(float)

def load_slope_file(file):
        #print(' -> Load', file)
    names = ['R','slope', 'err']
    slopes = np.genfromtxt(file, comments='#', dtype='float, float, float', names=names)
        # Create a dictionary (easier to use)
    model_dict = {}
    for name in names:
        model_dict[name] = slopes[name]
        
    df = pd.DataFrame(model_dict)

    return df


def calculate_name(df):
    # Perform calculation 
    log1 = np.log10(df['Rig'])
    log2 = np.log10(df['Flux'])


    rig, slope = [], []

    for i in range(0, len(log1) - 1):
        rig.append((df['Rig'][i+1]+df['Rig'][i]) / 2)

    for i in range(0, len(log1) - 1):
        slope.append((log2[i+1]-log2[i]) / (log1[i+1]-log1[i]))
        
    df_rig = pd.DataFrame(rig, columns=['Rig'])
    df_slope = pd.DataFrame(slope, columns=['Slope'])

    df_final = pd.concat([df_rig, df_slope], axis=1)
    df_final = df_final[df_final['Slope'].notna()]

    return df_final

def calcule_apha(NUCLEUS, i):
    # Alpha calculation
    alpha = 0
    if NUCLEUS[i] == 'H': alpha = 2.1        
    elif NUCLEUS[i] == 'HE': alpha = 2.3
    else: alpha = 2.34
    
    return alpha

def calcule_kappa(df):

    list_kappa = []

    knot=0.038904514
    eta_t=1
    delta = 0.51
    R_h=246.664
    delta_h=0.19
    s_h=4.073533e-02
    R_l = 4.53
    delta_l = -0.74
    s_l = 0.05
    ratio=(938.28+939.57)*1e-3

    for i in range(0, len(df)):
        beta = df['Rig'][i]/np.sqrt((df['Rig'][i]*df['Rig'][i])+(ratio*ratio))
        Kdiff = knot*pow(beta, eta_t)*pow(df['Rig'][i],delta)*pow(1.0+pow(df['Rig'][i]/R_h,(delta_h)/s_h),-s_h)*pow((1.0+pow(df['Rig'][i]/R_l,(delta_l-delta)/s_l)),s_l)
#        Kdiff = knot*pow(beta, eta_t)*pow(df['Rig'][i],delta)*pow(1.0+pow(df['Rig'][i]/R_h,(delta - delta_h)/s_h),-s_h)*pow((1.0+pow(df['Rig'][i]/R_l,(delta_l-delta)/s_l)),s_l)

        list_kappa.append(Kdiff)
        
    df_temp = pd.DataFrame(list_kappa, columns=['Kappa'])

    return df_temp

def calculate_derivative_kappa(df, df_kappa, nucleu):

    log1 = np.log10(df['Rig'])
    log2 = np.log10(df_kappa[nucleu + '_Kappa'])

    derivative = []

    for i in range(0, len(log1) - 1):
        derivative.append((log2[i+1]-log2[i]) / (log1[i+1]-log1[i]))
        
    df_derivative = pd.DataFrame(derivative, columns=['Derivative'])

    return df_derivative

def calculate_delta(df, df_nucleus, df_derivative, NUCLEUS):

    df_delta = pd.DataFrame(columns=NUCLEUS)

    for i in NUCLEUS:
        list_delta = []
        for j in range(0, len(df_nucleus)):

#            delta = df_nucleus[i+'_Slope'][j] 
            delta = (df_nucleus[i+'_Slope'][j] + (df['Alpha'][df['Nucleus'] == i] + df_derivative[i+'_Der_Kappa'][j])) / df_derivative[i+'_Der_Kappa'][j]
            
            
            list_delta.append(delta.values)
#            list_delta.append(delta)
        
        df_delta[i] = list_delta
        #df_delta[i] =df_delta[i].str.get(0)

    return df_delta
""""
def graphic(df_delta, df, df_input_slope_01, df_input_slope_02, df_input_slope_03, df_input_slope_04, df_input_slope_05, df_input_slope_06,df_input_slope_07, df_input_slope_08, df_input_slope_09, df_input_slope_10, df_input_slope_11, df_input_slope_12, df_input_slope_13, df_input_slope_14, df_input_slope_15):#

    df_graphic = pd.DataFrame()

    df_graphic['X'] = df.index+1
    df_graphic['Y1'] = df_delta.iloc[500].values# 20 GV
    df_graphic['Y2'] = df_delta.iloc[715].values# 200 GV
    df_graphic['Y3'] = df_delta.iloc[930].values#2 TV
#    df_graphic['Y4'] = df_delta.iloc[1361].values#200 TV
#    df_graphic['Y5'] = df_delta.iloc[1576].values#2 PV
#    df_graphic['Y6'] = df_delta.iloc[].values



#    fig, ax = plt.subplots()


#    ax.errorbar(1, df_input_slope_01['slope'].iloc[2],
 #           xerr=0,
 #           yerr=df_input_slope_01['err'].iloc[2],
 #           fmt='-o')


#    plt.plot(df_graphic['X'], df_graphic['Y2'], '-', color="purple",label = 'SLIM model (200 GV)')
#    plt.plot(1, df_input_slope_01['slope'].iloc[9],'.',color="blue",label = 'AMS data (200 GV)')
#    plt.plot(2, df_input_slope_02['slope'].iloc[9],'.', color="blue")
#    plt.plot(3, df_input_slope_03['slope'].iloc[9],'.', color="blue")
     
#    plt.plot(4, df_input_slope_04['slope'].iloc[9],'.', color="blue")
#    plt.plot(5, df_input_slope_05['slope'].iloc[9],'.', color="blue")
#    plt.plot(6, df_input_slope_06['slope'].iloc[9],'.', color="blue")
#    plt.plot(7, df_input_slope_07['slope'].iloc[9],'.', color="blue")
#    plt.plot(8, df_input_slope_08['slope'].iloc[9],'.', color="blue")
#    plt.plot(9, df_input_slope_09['slope'].iloc[9],'.', color="blue")
#    plt.plot(10, df_input_slope_10['slope'].iloc[9],'.', color="blue")
#    plt.plot(11, df_input_slope_11['slope'].iloc[9],'.', color="blue")
#    plt.plot(12, df_input_slope_12['slope'].iloc[9],'.', color="blue")
#    plt.plot(13, df_input_slope_13['slope'].iloc[9],'.', color="blue")
#    plt.plot(14, df_input_slope_14['slope'].iloc[9],'.',color="blue")
#    plt.plot(26, df_input_slope_15['slope'].iloc[9],'.',color="blue")

#    slope20GV=np.array([0.997424, 0.8779452, 0,0,0,0.8304225,0.3030398,0.944165,0.01098553,0.7367114,0.03243083,0.8466638,0.3899467,0.915608,0.1671138,0.6456518,0.01122617,0.08675553,0.01120044, 0.2776916, 0.01128176,0.01116885,0.01100264,0.01094757,0.08957103,0.9564957 ])

#    plt.plot(1, df_input_slope_01['slope'].iloc[2],'.',color="blue")
    plt.plot(2, df_input_slope_02['slope'].iloc[2],'.', color="blue",label = 'AMS data (20 GV)')
#    plt.plot(3, df_input_slope_03['slope'].iloc[2],'.', color="blue")
     
#    plt.plot(4, df_input_slope_04['slope'].iloc[2],'.', color="blue")
#    plt.plot(5, df_input_slope_05['slope'].iloc[2],'.', color="blue")
#    plt.plot(6, df_input_slope_06['slope'].iloc[2],'.', color="blue")
#    plt.plot(7, df_input_slope_07['slope'].iloc[2],'.', color="blue")
#    plt.plot(8, df_input_slope_08['slope'].iloc[2],'.', color="blue")
#    plt.plot(9, df_input_slope_09['slope'].iloc[2],'.', color="blue")
#    plt.plot(10, df_input_slope_10['slope'].iloc[2],'.', color="blue")
#    plt.plot(11, df_input_slope_11['slope'].iloc[2],'.', color="blue")
#    plt.plot(12, df_input_slope_12['slope'].iloc[2],'.', color="blue")
#    plt.plot(13, df_input_slope_13['slope'].iloc[2],'.', color="blue")
#    plt.plot(14, df_input_slope_14['slope'].iloc[2],'.',color="blue")
#    plt.plot(26, df_input_slope_15['slope'].iloc[2],'.',color="blue")

 #   plt.tick_params(which='both', direction='in')    
 
    
    plt.xlabel('Atomic number Z', fontsize=14)
    plt.ylabel('${\gamma_{obs}}$', fontsize=14)
    plt.legend()



    plt.savefig("results/Figure4_Pedroplot.png", dpi=500)
    plt.close()
   
 """   
###########
def graphic_02(df_delta, df):
    # Builds subplots
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)   # Remove horizontal space between axes

#    frac50GV = np.array([1., 0.9177224,0,0,0,0.8753395,0.3207792, 0.9676795, 0.1493617,  0.8148991, 0.1884535,0.8944432,0.4464421,0.9468646,0.3082725,0.7653568,0.1480221,0.3154043,0.1611675,0.5004733,0.1494211,0.1459963, 0.1397496,0.2021739,0.3136033,0.9678505])
    frac2PV = np.array([0.9999897,0.9974062, 0,0,0,0.9949234,0.9272684,0.9988596,0.2930064, 0.9919513,0.8587632,0.9957776,0.9509986,0.9981942,0.9273641,0.9906267,0.8307454,0.9387634,0.3187681,0.9627502,0.2780304,0.265079,0.2397904,0.8438472,0.894062, 0.9984654])
#        0.9999657,0.9945096,0,0,0,0.98952,0.8604076,0.9976326,0.1667536,0.9834443,0.7462818,0.9912903,0.9039835,0.9962376,0.860122,0.9805711,0.7012834,0.8802465,0.1834739,0.9258404,0.1568956,0.1487019,0.1331139,0.7253163,0.8053674,0.9968783 ])
    frac200TV = np.array([0.9999657,0.9945096,0,0,0,0.98952,0.8604076,0.9976326,0.1667536,0.9834443,0.7462818,0.9912903,0.9039835,0.9962376,0.860122,0.9805711,0.7012834,0.8802465,0.1834739,0.9258404,0.1568956,0.1487019,0.1331139,0.7253163,0.8053674,0.9968783 ])
    frac20GV=np.array([0.997424, 0.8779452, 0,0,0,0.8304225,0.3030398,0.944165,0.01098553,0.7367114,0.03243083,0.8466638,0.3899467,0.915608,0.1671138,0.6456518,0.01122617,0.08675553,0.01120044, 0.2776916, 0.01128176,0.01116885,0.01100264,0.01094757,0.08957103,0.9564957 ])
    #speciesZ = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    elements = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe']
    w20GV= [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    w200TV= [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]

    axs[0].bar(elements, frac2PV,color="magenta",label = '2 PV') 
    axs[0].bar(elements, frac200TV,width =w200TV,color="orange",label = '200 TV')     
    axs[0].bar(elements, frac20GV,width =w20GV,color="darkblue",label = '20 GV')
    
    axs[0].tick_params(which='both', direction='in')    
    axs[0].xaxis.set_ticks_position('both')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].xaxis.set_major_locator(plt.MultipleLocator(2))
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)    
    
    axs[0].set_ylim(0.01, 1.2)
    axs[0].set_ylabel('Primary fraction [%]', fontsize=14)

    trans = mtransforms.ScaledTranslation(10/62, -40/62, fig.dpi_scale_trans)
    axs[0].text(0.0, 1.1, 'a)', transform=axs[0].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='white', pad=-5.0))

    handles, labels = axs[0].get_legend_handles_labels()
#    axs[0].legend(loc="upper right", fontsize=14, ncol=3, handles[::-1], labels[::-1])
    axs[0].legend(handles[::-1], labels[::-1],fontsize=14, ncol=3)
#       axs[0].legend(handles[::-1], labels[::-1])

#    axs[1] = axs[0].twiny()   
    
    
    # Plot 02##################################
    
    df_graphic = pd.DataFrame()

    #df_graphic['X'] = df.index+1
    elem = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

    #elements = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe']
 
    df_graphic['Y1'] = df_delta.iloc[500].values# 20 GV
    df_graphic['Y2'] = df_delta.iloc[715].values# 200 GV
    df_graphic['Y3'] = df_delta.iloc[930].values#2 TV
    df_graphic['Y4'] = df_delta.iloc[1361].values#200 TV
    df_graphic['Y5'] = df_delta.iloc[1576].values#2 PV

    plt.plot(elem, df_graphic['Y1'], color="darkblue", linewidth=2.5,label = '20 GV')
    plt.plot(elem, df_graphic['Y2'], color="blue", linestyle = 'dotted', linewidth=2.5,label = '200 GV')
    plt.plot(elem, df_graphic['Y3'], color="green", linestyle = 'dashed', linewidth=2.5,label = '2 TV')
    plt.plot(elem, df_graphic['Y4'], color="orange", linestyle = 'dotted', linewidth=2.5,label = '200 TV')
    plt.plot(elem, df_graphic['Y5'], color="magenta", linewidth=2.5,label = '2 PV')

    
    axs[1].tick_params(which='both', direction='in')
    axs[1].xaxis.set_ticks_position('both')
    axs[1].yaxis.set_ticks_position('both')
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(2))
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
    axs[1].yaxis.set_major_locator(plt.MultipleLocator(0.2))
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)    
    
    axs[1].set_ylim(-1.05, 1.15)
    
    trans = mtransforms.ScaledTranslation(10/62, -10/62, fig.dpi_scale_trans)
    
    axs[1].text(0.0, 1.0, 'b)', transform=axs[1].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='white', pad=-5.0))
    
    plt.xlabel('Atomic number Z', fontsize=16)
    plt.ylabel('$\Delta_{\gamma}$', fontsize=16)
    plt.legend(loc="upper right", fontsize=14, ncol=5)
#    plt.tight_layout()

    plt.savefig("results/Figure2_delta_vs_Z.png", dpi=500)
    plt.close()
    
    
    

    
    
#####################################################################


if (__name__ == "__main__"):

    NUCLEUS = ['H','HE','LI','BE','B','C','N','O','F','NE','NA','MG','AL','SI','P','S','CL','AR','K','CA','SC','TI','V','CR','MN','FE']
    #NUCLEUS = ['H']

    df = pd.DataFrame(NUCLEUS, columns = ['Nucleus'])

    df['Alpha'] = 0

    df_nucleus = pd.DataFrame()
    df_kappa = pd.DataFrame()
    df_derivative = pd.DataFrame()

    for i in range(len(NUCLEUS)):
        df_input = pd.read_csv('/Users/mvecchi/USINE/output/2kevents/SLIM/local_fluxes_'+NUCLEUS[i]+'_R_Model1DKisoVc_SolMod0DFF_phi0_670GV.out', sep='\t')
#        df_input = pd.read_csv('/Users/mvecchi/USINE/output/SLIM/allspecies/local_fluxes_'+NUCLEUS[i]+'_R_Model1DKisoVc_SolMod0DFF_phi0_670GV.out', sep='\t')

        df_input_slope_01 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z01_H_AMS02.txt')
        df_input_slope_02 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z02_He_AMS02.txt')      
        df_input_slope_03 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z03_Li_AMS02.txt')
        df_input_slope_04 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z04_Be_AMS02.txt')      
        df_input_slope_05 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z05_B_AMS02.txt')      

        df_input_slope_06 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z06_C_AMS02.txt')
        df_input_slope_07 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z07_N_AMS02.txt')      
        df_input_slope_08 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z08_O_AMS02.txt')
        df_input_slope_09 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z09_F_AMS02.txt')      
        df_input_slope_10 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z10_Ne_AMS02.txt')      

        df_input_slope_11 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z11_Na_AMS02.txt')
        df_input_slope_12 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z12_Mg_AMS02.txt')      
        df_input_slope_13 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z13_Al_AMS02.txt')
        df_input_slope_14 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z14_Si_AMS02.txt')      
        df_input_slope_15 = load_slope_file('/Users/mvecchi/Documents/Work/Research/CRAC/CRACpedroReload/newcode/slope_AMSdata/slope_Z26_Fe_AMS02.txt')      


        df_input = handle_input_file(df_input)

        df['Alpha'].iloc[i] = calcule_apha(NUCLEUS, i)

        df_temp = calculate_name(df_input)

        df_nucleus[NUCLEUS[i]+'_Rig'] = df_temp['Rig']
        df_nucleus[NUCLEUS[i]+'_Slope'] = df_temp['Slope']

        df_temp = calcule_kappa(df_input)
        df_kappa[NUCLEUS[i]+'_Rig'] = df_input['Rig']
        df_kappa[NUCLEUS[i]+'_Kappa'] = df_temp['Kappa']

        df_temp = calculate_derivative_kappa(df_input, df_kappa, NUCLEUS[i])
        df_derivative[NUCLEUS[i]+'_Der_Kappa'] = df_temp['Derivative']

    df_delta = calculate_delta(df, df_nucleus, df_derivative, NUCLEUS)

#    print(df_input_slope_01)

#    graphic(df_delta, df, df_input_slope_01, df_input_slope_02,df_input_slope_03, df_input_slope_04, df_input_slope_05, df_input_slope_06, df_input_slope_07, df_input_slope_08, df_input_slope_09, df_input_slope_10, df_input_slope_11, df_input_slope_12, df_input_slope_13, df_input_slope_14, df_input_slope_15)
    graphic_02(df_delta, df)

print('OK')






