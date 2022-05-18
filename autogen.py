
import os
from re import S
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d

data_dir = '/home/felix/USINE_docker/data/USINE_FILES'


# Size scaling, figsize is given in inches so we need to scale it to the width we require for our paper
# An A4 papers dimensions are 21cm x 29.7cm so:
cm =  1/2.54 
margin = 2.5*cm
pt_inch = 1/72.27
column_sep = 20*pt_inch
twocolumn_width = (21*cm - 2*margin)*3
column_width = (21*cm - 2*margin - column_sep)

# Set default plt params
plt.rc('font', size=10)
plt.rc('font', family='serif')
plt.rc('font', serif="Times New Roman")
plt.rc('figure', titlesize=14) # TODO: Change axis label size to something larger then decrease tick size and see if it fits that way


latex_format = {'Va':'V_{a}', 'Vc':'V_c', 'K0':'K_0', 'delta':'\delta', 'eta_t':'\eta_t', 'Rbreak':'R_{break}','Deltabreak':'\delta_{break}','sbreak':'s_{break}','Rlow':'R_{low}', 'deltalow':'\delta_{low}', 'slow':'s_{low}'}
name_dict = {'BO': 'B/O', 'BeC': 'Be/C', 'LiO': 'Li/O', 'LiC': 'Li/C', 'FSi': 'F/Si', 'BeO': 'Be/O', 'C': 'C', 'HeO': 'He/O', '1HBAR': '1HBAR', 'Ne': 'Ne', 'CO': 'C/O', 'He': 'He', 'BeB': 'Be/B', 'SiO': 'Si/O', 'N': 'N', 'O': 'O', 'NeO': 'Ne/O', 'Si': 'Si', 'B': 'B', 'LiB': 'Li/B', 'H': 'H', 'MgO': 'Mg/O', 'Mg': 'Mg', 'BC': 'B/C','Be':'Be','Mn':'Mn','F':'F','Ca':'Ca','Ti':'Ti', 'Li': 'Li','Cr':'Cr'}
ams_name_dict = {'BEC':'BeC','AL':'Al','FE':'Fe','CL':'Cl','V':'V','S':'S','AR':'Ar','P':'P','K':'K','NA':'Na','SC':'Sc','BO':'BO','C': 'C','LIO':'LiO', 'LIC':'LiC', 'FSI':'FSi', 'BEO':'BeO', 'HEO':'HeO', '1HBAR':'1HBAR', 'NE':'Ne', 'CO':'CO', 'HE':'He', 'BEB':'BeB', 'SIO':'SiO', 'N':'N', 'O':'O', 'NEO':'NeO', 'SI':'Si', 'B':'B', 'LIB':'LiB', 'H':'H', 'MGO':'MgO', 'MG':'Mg', 'BC':'BC', 'LI':'Li','CR':'Cr','TI':'Ti','BE':'Be','MN':'Mn','CA':'Ca','F':'F'}
element_dict = {'H':1,'He':2, 'Li':3, 'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26}


def get_transport_and_source(model_dir):
    transport_params = {}
    with open('/home/felix/USINE_docker/'+model_dir+'/model_vals_pars.out', 'r') as f:
        lines = f.readlines()
        cond = False
    for line in lines:
        if '# Source' in line:
            cond = False
            break 
        if cond and line!='\n':
            key, value = line.split('=')
            # Latex format key
            key = latex_format[key.strip(' ')]
            # Add units
            key = key + ' (' + value.split('[')[1].split(']')[0] + ')'
            if key.strip(' ') not in transport_params.keys():   
                transport_params[key.strip(' ')] = float(value.split('[')[0].strip(' '))
        if '# Transport' in line:
            cond = True
    # Read .C files for Double ts to retrieve data for isotopes etc. 
    C_dat = {}
    for i in os.listdir(model_dir):
        if i.endswith('.C'):
            with open(os.path.join(model_dir, i), 'r') as f:
                lines = f.readlines()
            cond = False
            for line in lines:
                # If stop not found
                if cond:
                    if '}' not in line:
                        content.append(float(line.strip('\n').strip(',').strip(';').strip('')))
                    else:
                        content.append(float(line.strip('\n').strip(',').strip(';').strip('')[:-1:]))
                # Recognize start
                if 'Double_t' in line:
                    name = line[12::].split('[')[0] # Gives data descriptor
                    content = []
                    cond = True
                # Find stop
                if "}" in line and cond:
                    cond = False
                    C_dat[name] = content
    
    AMS_data = {} # TODO: Load and format ams data for plotting

    for i in os.listdir(data_dir):
        if 'orig' not in i and 'cov' not in i and 'prelim' not in i and "DAMPE" not in i and 'CALET' not in i and 'BC_AMS02.USINE' not in i and '.DS' not in i:
            with open(os.path.join(data_dir, i), 'r') as f:
                try:
                    lines = f.readlines()
                except:
                    print(os.path.join(data_dir, i))
                    pass
            commented_lines = len([i for i in lines if '#' in i])
            lines = [i for i in lines if '#   Col.' in i]
            headers = [i.split('-')[1].strip('\n').strip(' ') for i in lines]
            # FSi O and some more dont have headers, the assumption is they all have the same headers
            if len(headers)<2: 
                print(i.split('_')[0], ' doesnt have headers')
                headers = headers_backup
            else:
                headers_backup = headers
            try:
                #if i.split('_')[0]=='O': print(pd.read_csv(os.path.join(data_dir, i), names=headers, skiprows=commented_lines,encoding='us-ascii', delimiter=r"\s+"))
                AMS_data[i.split('_')[0]]=pd.read_csv(os.path.join(data_dir, i), names=headers, skiprows=commented_lines,encoding='us-ascii', delimiter=r"\s+")
            except Exception as e:
                print(e)
                print(os.path.join(data_dir, i))
    
    source_params = {}

    model_val_files ={}
    for i in os.listdir('/home/felix/USINE_docker/'+model_dir):
        if 'out' in i:
            model_val_files['_'.join(i.split('_')[:-1:])]=os.path.join('/home/felix/USINE_docker/',model_dir,'model_vals_pars.out')


    # Extract Source parameters
    for i in model_val_files:
        with open(model_val_files[i], 'r') as f:
            lines = f.readlines()
            cond = False
        for line in lines:
            if cond and line!='\n':
                key, value = line.split('=')
                # Add units
                key = key + ' (' + value.split('[')[1].split(']')[0] + ')'
                if key.strip(' ') not in source_params.keys():   
                    source_params[key.strip(' ')] = [float(value.split('[')[0].strip(' '))]
                else:
                    source_params[key.strip(' ')].append(float(value.split('[')[0].strip(' ')))
            if '# Source' in line:
                cond = True

    return transport_params, source_params, AMS_data, C_dat


def make_B_C_BC_plot(AMS_data, C_dat, model_dir):
    fig = plt.figure(figsize=(twocolumn_width,0.5*twocolumn_width))
    fig.set_dpi(72*2)

    plt.subplots_adjust(hspace=0.4)

    #       Plotting
    ax1 = fig.add_subplot(4,2,1)
    ax1_r = fig.add_subplot(4,2,3)
    ax2 = fig.add_subplot(4,2,2)
    ax2_r = fig.add_subplot(4,2,4)
    ax3 = fig.add_subplot(4,2,(5,6))
    ax3_r = fig.add_subplot(4,2,(7,8))
    for i in (ax1, ax2, ax3):
        i.set_xscale('log')
        i.set_xlabel('$R\ [GV]$')
        i.grid()

    for i in (ax1_r, ax2_r, ax3_r):
        i.set_xscale('log')
        i.set_ylabel('Residual')
        i.set_xlabel('$R\ [GV]$')
        i.grid()

    for i in (ax1, ax2):
        i.set_ylabel('$\Phi\ [GV\ m^2\ s\ sr^{-1}]$')

    ax3.set_ylabel('B/C')

    for key in C_dat:
        if 'model_' in key:
            if 'fy' not in key:
                label = key.split('_')[1]
                if label == 'B':
                    data = AMS_data[key.split('_')[1]]
                    # Make interpolation to fit to AMS data
                    f=interp1d(C_dat[key], C_dat[key.replace('fx','fy')], kind='cubic')
                    interpolated_fit = f(data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'])
                    # Get coloring scheme
                    colors = [data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit-(data['ERR_STAT']+data['ERR_SYST']), data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit+(data['ERR_STAT+']+data['ERR_SYST+'])]
                    colors = ['black' if i<0<j else 'red' for i,j in zip(colors[0],colors[1])]
                    # Plot Graph
                    ax1.plot(C_dat[key], C_dat[key.replace('fx','fy')], label='modeled '+label)
                    ax1.errorbar(y=data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio'], x=data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'],yerr=[data['ERR_STAT']+data['ERR_SYST'], data['ERR_STAT+']+data['ERR_SYST+']] ,label='AMS data '+label,fmt='None',c='black',ecolor=colors)
                    ax1.set_title('B flux')
                    # Plot residuals
                    ax1_r.errorbar(y=data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit, x=data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'], yerr=[data['ERR_STAT']+data['ERR_SYST'], data['ERR_STAT+']+data['ERR_SYST+']] ,label='AMS data '+label,fmt='None',c='black',ecolor=colors)
                if label == 'C':
                    data = AMS_data[key.split('_')[1]]
                    # Make interpolation to fit to AMS data
                    f=interp1d(C_dat[key], C_dat[key.replace('fx','fy')], kind='cubic')
                    interpolated_fit = f(data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'])
                    # Get coloring scheme
                    colors = [data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit-(data['ERR_STAT']+data['ERR_SYST']), data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit+(data['ERR_STAT+']+data['ERR_SYST+'])]
                    colors = ['black' if i<0<j else 'red' for i,j in zip(colors[0],colors[1])]
                    # Plot Graph
                    ax2.plot(C_dat[key], C_dat[key.replace('fx','fy')], label='modeled '+label)
                    ax2.errorbar(y=data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio'], x=data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'],label='AMS data '+label,fmt='None',c='black',ecolor=colors)
                    ax2.set_title('C flux')
                    # Plot residuals
                    ax2_r.errorbar(y=data['QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio']-interpolated_fit, x=data[ '<E>: mean value bin [GeV/n, GeV, GV, or GeV]'], yerr=[data['ERR_STAT']+data['ERR_SYST'], data['ERR_STAT+']+data['ERR_SYST+']] ,label='AMS data '+label,fmt='None',c='black',ecolor=colors)
                if label == 'BC':
                    # Make interpolation to fit to AMS data for residuals
                    name = [i for i in C_dat.keys() if 'data_' in i and 'fx' in i and label in i][0]
                    f=interp1d(C_dat[key], C_dat[key.replace('fx','fy')], kind='cubic')
                    interpolated_fit = f(C_dat[name])
                    # Get coloring scheme
                    colors = [C_dat[name.replace('fx','fy')]-interpolated_fit-C_dat[name.replace('fx','fely')], C_dat[name.replace('fx','fy')]-interpolated_fit+C_dat[name.replace('fx','fehy')]]
                    colors = ['black' if i<0<j else 'red' for i,j in zip(colors[0],colors[1])]
                    # Plot Graph
                    ax3.set_title('B/C ratio')
                    ax3.plot(C_dat[key], C_dat[key.replace('fx','fy')], label=label)
                    # Get AMS data and plot
                    if label == 'BC': label = 'B/C'
                    ax3.errorbar(x=C_dat[name], y=C_dat[name.replace('fx','fy')],yerr= [C_dat[name.replace('fx','fely')], C_dat[name.replace('fx','fehy')]],label=label+' AMS data',fmt='None',c='black',ecolor=colors)
                    # Plot residuals
                    ax3_r.errorbar(x=C_dat[name], y=C_dat[name.replace('fx','fy')]-interpolated_fit,yerr= [C_dat[name.replace('fx','fely')], C_dat[name.replace('fx','fehy')]],label=label+' AMS data',fmt='None',ecolor=colors)


    for i in (ax1, ax2, ax3,ax1_r, ax2_r, ax3_r):
        i.axvline(transport_params['R_{low} (GV)'],label='$R_{low}$',c='green')
        i.axvline(transport_params['R_{break} (GV)'],label='$R_{high}$',c='orange')
    for i in (ax1, ax2, ax3):
        i.legend()
    for i in ((ax1,ax1_r), (ax2,ax2_r), (ax3, ax3_r)):
        i[0].set_xlim(i[1].get_xlim())

    plt.savefig('/home/felix/USINE_docker/'+model_dir+'/plots/BC.png',bbox_inches='tight')

def calculate_diff_coeff(Rigidity, transport_params):
    """
    Calculate the diffusion coefficient as a function of rigidity
    Rigidity - list of rigidity values
    model_val_files_dict - dictionary of model values from model_vals_pars.out loaded as below
    """
    for key in transport_params:
        if type(transport_params[key]) == list:
            transport_params[key] = transport_params[key][0]
    ratio = (938.28+939.57)*1e-3
    # Relativistic velocity
    beta = Rigidity/np.sqrt(Rigidity**2+ratio**2)
    # Calculate the diffusion coefficient
    nonR_R = transport_params['K_0 (kpc^2/Myr)']*np.power(beta, transport_params['\\eta_t (-)'])
    low_R = np.power(1+np.power(Rigidity/transport_params['R_{low} (GV)'], (transport_params['\\delta_{low} (-)']-transport_params['\\delta (-)'])/transport_params['s_{low} (-)']),transport_params['s_{low} (-)'])
    intermid_R = np.power(Rigidity, transport_params['\\delta (-)'])
    high_R = np.power(1+np.power(Rigidity/transport_params['R_{break} (GV)'], (transport_params['\\delta_{break} (-)'])/transport_params['s_{break} (-)']),-transport_params['s_{break} (-)'])
    K = nonR_R*low_R*intermid_R*high_R
    return K

def get_loglog_slope(fx, fy):
    '''fx and fy as returned by model
    returns log slopes
    '''
    logx = np.log10(fx)
    logy = np.log10(fy)
    grad = (logy[1::]-logy[:-1:])/(logx[1::]-logx[:-1:])
    return grad

def make_diff_plot(transport_params):
    R = np.logspace(-1,6,1000)
    K = calculate_diff_coeff(R, transport_params)

    fig = plt.figure(figsize=(10,10))

    plt.xscale('log')

    plt.ylabel(r'$K(R)$')
    plt.xlabel(r'$R\ (GV)$')
    plt.title(r'Diffusion coefficient against Rigidity') 

    plt.plot(R,K)

    plt.grid()
    plt.legend()

    plt.savefig('/home/felix/USINE_docker/'+model_dir+'/plots/diff_slope.png',bbox_inches='tight')


def get_model_slopes(C_dat):
    SLIM_slopes = {}
    for key in C_dat.keys():
        if 'fx' not in key and 'model' in key:
            SLIM_slopes[key.split('_')[1]] = [C_dat[key.replace('fy','fx')][1::], get_loglog_slope(C_dat[key.replace('fy','fx')],C_dat[key])]
    return SLIM_slopes


# TODO: THis requires modification once all are there
def plot_all_species_flux_slopes(SLIM_slopes):
    fig, ax = plt.subplots(5,2,figsize=(twocolumn_width,2*twocolumn_width))
    #fig.set_dpi(72*2)

    # To avoid overwriting the original instance
    axes = ax.flatten()
    plt.subplots_adjust(hspace=0.4)

    for i in axes:
        i.set_xscale('log')
        i.set_xlabel('$R\ [GV]$')
        i.grid()
    print(SLIM_slopes.keys())
    ax_counter = 0
    for key in SLIM_slopes:
        # Get correct casing for AMS indexing
        ams_name = ams_name_dict[key]
        # Add fraction / where required
        name = name_dict[ams_name]

        if '/' not in name: 

            axes[ax_counter].set_ylabel(r'$S_'+'\x7B'+'{}'.format(name)+'\x7D$')
        
            axes[ax_counter].plot(*SLIM_slopes[key], label='SLIM')

            axes[ax_counter].legend()
        
            axes[ax_counter].set_title(name)
        
            #axes[ax_counter].set_ylim(-5,5)
        
            ax_counter+=1
    
    plt.savefig('/home/felix/USINE_docker/'+model_dir+'/plots/flux_slopes.png',bbox_inches='tight')


def make_slope_Z_plot(SLIM_slopes):
    flux_20 = {}
    flux_200 = {}

    for key in SLIM_slopes:
        interpolator = interp1d(SLIM_slopes[key][0],SLIM_slopes[key][1],kind='cubic')
        flux_20[ams_name_dict[key]] = interpolator(20)
        flux_200[ams_name_dict[key]] = interpolator(200)
    
    fig,ax = plt.subplots(1,1,figsize=(twocolumn_width,column_width))

    ax.set_ylabel('$S_\x7B \phi \x7D$')
    ax.set_xlabel('Atomic Number $Z$')

    ax.set_xticks(np.arange(0,27,1))
    #ax.set_xticklabels(np.arange(0,27,1))
    ax.set_xlim(1,26)

    x2 = ax.twiny()

    x2.set_xticks(np.arange(0,27,1))
    #x2.set_xticks([i for i in element_dict.values()])
    x2.set_xticklabels(['']+[i for i in element_dict])

    flux20 = np.zeros(len(element_dict))
    flux200 = np.zeros(len(element_dict))

    for key in flux_20:
        if key in element_dict:
            flux20[element_dict[key]-1] = flux_20[key]
            flux200[element_dict[key]-1] = flux_200[key]

    plt.plot(np.arange(1,27,1),flux20,label='20 GV')
    plt.plot(np.arange(1,27,1),flux200,label='200 GV')
    plt.xlim(1,26)
    plt.grid()
    plt.legend()
    plt.title(r"$\phi_{FF}$=0."+model_dir.split('_')[-1])
    plt.savefig('/home/felix/USINE_docker/'+model_dir+'/plots/Zslope.png',bbox_inches='tight')


if __name__=='__main__':
    # One model plotting
    # Load the data
    model_dir = 'USINE_out_97'

    transport_params, source_params, AMS_data, C_dat = get_transport_and_source(model_dir)

    SLIM_slopes = get_model_slopes(C_dat)
    if not os.path.isdir(model_dir+'/plots'):
        os.mkdir(model_dir+'/plots')

    make_B_C_BC_plot(AMS_data, C_dat, model_dir)

    make_diff_plot(transport_params)

    #plot_all_species_flux_slopes(SLIM_slopes)

    make_slope_Z_plot(SLIM_slopes)


