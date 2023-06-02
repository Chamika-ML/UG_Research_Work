# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:39:29 2023

@author: Anuruddha
"""

import numpy as np
import pickle 
import  streamlit as st
from streamlit_option_menu import option_menu
import lightkurve as lk
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt



# functions
def plot(x,y):
    fg,ax = plt.subplots(1,1)
    ax.plot(x,y)
    ax.set_title(f'The light curve of {KIC_No}')
    ax.set_xlabel("time (days)")
    ax.set_ylabel("flux ($\mathregular{es^{-1}}$)")
    fg.tight_layout()
    st.pyplot(fg)
    
    
def Chi_square(flux,err):
    
    mean = np.mean(flux)
    
    chi_square = np.sum((flux-mean)**2/err**2)
    
    return chi_square



def Robust_median_statistic(flux,err):
    
    N = len(flux)
    median = np.median(flux)
    
    ROMS = (np.sum(abs(flux - median)/err))/(N-1)
    
    return ROMS



def Von_Neumann_ratio(flux):
    
    N = len(flux)
    mean = np.mean(flux)
    var = np.sum((flux - mean)**2/(N-1))
    delta_sq = 0
    
    for i in range(len(flux) - 1):
        
        flux_1 = flux[i]
        flux_2 = flux[i+1]
        
        delta_sq +=(flux_2 - flux_1)**2/(N-1)
    
    VNR = delta_sq/var
    
    return VNR
        
        
        

def Slop_Vector(time,flux):
    
    """calculate slope between two points. slope = distence_between_two_flux_points/distance_between_two_time_points
    retuens slope vector which contain slope between every two points."""
    
    flux_diff_vec = []
    time_diff_vec = []

    for i in range(len(flux)-1):
        
        flux_diff = flux[i+1] - flux[i]
        time_diff = time[i+1] - time[i]
    
        flux_diff_vec.append(flux_diff)
        time_diff_vec.append(time_diff)
    

    flux_diff_vec = np.array(flux_diff_vec)
    time_diff_vec = np.array(time_diff_vec)
    
    slope_vec = flux_diff_vec/time_diff_vec
    
    return slope_vec


def Peak_to_peak_variability(flux,err):
    
    fulx_min_sigma = []
    fulx_pls_sigma = []
    
    for i in range(len(flux)):
        
        f_m_s = flux[i] - err[i]
        f_p_s = flux[i] + err[i]
        
        fulx_min_sigma.append(f_m_s)
        fulx_pls_sigma.append(f_p_s)
        
    max_fms = max(fulx_min_sigma)
    min_fps = min(fulx_pls_sigma)
    
    v = (max_fms - min_fps)/(max_fms + min_fps)
    
    return v
    

def Lag_1_autocorrelation(flux):
    
    mean = np.mean(flux)
    numa = 0
    dino = np.sum((flux - mean)**2)
    
    for i in range(1,len(flux)):
        
        numa += (flux[i] - mean)*(flux[i-1] - mean)
    
    L1AC = numa/dino
    
    return L1AC



def fourier_comp2(time,flux):
    
    N = len(flux)
    T = time[1] - time[0]

    yf = np.fft.fft(flux)
    xf = np.fft.fftfreq(N,T)[1:N//2]
    
    max_freq_index = np.argwhere((np.abs(yf[1:N//2]) == np.max(np.abs(yf[1:N//2]))))
    max_freq = xf[max_freq_index]
    f1 = max_freq[0][0] 
    
    second_max_freq_index = np.argwhere((np.abs(yf[1:N//2])==np.sort((np.abs(yf[1:N//2])))[::-1][1]))
    second_max_freq = xf[second_max_freq_index]
    f2 = second_max_freq[0][0]
    
    
    return f1,f2


def fourier_comp(time,flux):
    
    N = len(flux)
    T = time[1] - time[0]

    yf = np.fft.fft(flux)
    xf = np.fft.fftfreq(N,T)[1:N//2]
    
    max_freq_index = np.argwhere((np.abs(yf[1:N//2]) == np.max(np.abs(yf[1:N//2]))))
    max_freq = xf[max_freq_index]
    f1 = max_freq[0][0] 
    
    return f1


def feature_genarate1(time,flux,err):
    
    
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features
    
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        mode = stat.mode(flux_vec)[0][0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))

              
        # stat features (using functions)
        
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)


        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
        
        
        # append features to a list

        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))
        
        
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))

      
        
        feature_list.append(features)
    

    return feature_list       


def feature_genarate2(time,flux,err):
    
    
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features
    
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        mode = stat.mode(flux_vec)[0][0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))
        
        # stat features (using functions)
        
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)


        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
        
        fc1,fc2 = fourier_comp2(time_vec,flux_vec)
        
        
        # append features to a list

        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))
        
        
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))
        
        features.append(round(fc1,4))
        features.append(round(fc2,4))
       
        
        feature_list.append(features)
    

    return feature_list



def feature_genarate3(time,flux,err):
    
    
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features
    
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        mode = stat.mode(flux_vec)[0][0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))

              
        # stat features (using functions)
        
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)


        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
        
        fc = fourier_comp(time_vec,flux_vec)
        
        
        # append features to a list

        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))
        
        
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))
        
        features.append(round(fc,4))
        

      
        
        feature_list.append(features)
    

    return feature_list

# loading saved models

c1_model = pickle.load(open("classifier_1.sav",'rb'))
c2_model = pickle.load(open("classifier_2.sav",'rb'))
c3_model = pickle.load(open("classifier_3.sav",'rb'))


# create side bar or navigation

with st.sidebar:
    
    selected = option_menu('Variable Star Classification System',
                           ['Main Prediction', 'Pulsation stars'],
                           
                           icons = ['stars','heart-pulse-fill'], # icons from bootstrap website
                           
                           default_index = 0) # default_index is defalut selected page




# Main Prediction page

if (selected == 'Main Prediction'):
    
    # page title
    st.title('Pulsation, Pure Binary and Binary + Pulsation Prediction')
    
    
    # layout of the page
    # make three columns in one line and place three textboxes
    # these order must be our training data set column order
    # these are page text boxes
   
    col1,col2 = st.columns(2)
    
    with col1:
        KIC_No = st.text_input('KIC Number')
        
    
    # code for Prediction 
    
    final_reult = '' # final result
    
    
    # creating a button for prediction
    
    if st.button('Type of  the star'):
        
        lc  = lk.search_lightcurve( KIC_No, author = 'Kepler', cadence = 'long', quarter = 9).download()
        lc.plot()
        # extracrt dat 

        time_t = np.array(lc.time.to_value('jd'))
        time_t = time_t - time_t[0]
        flux_t = np.array(lc.flux.to_value())
        err_t = np.array(lc.flux_err.to_value())
        
        time_t = time_t.tolist()
        flux_t = flux_t.tolist()
        err_t = err_t.tolist()
        
        # create data frame
        
        data = {'time':time_t , 'flux':flux_t, 'err':err_t}
        df = pd.DataFrame(data)
        
        # fill nans using backword (main) and forward( if last point NaN) fill methods.
        
        new_df_back = df.fillna(method='bfill')
        new_df = new_df_back.fillna(method='ffill')
        
        
        # predict 
        
        time_p = np.array([new_df.time.tolist()])
        flux_p = np.array([new_df.flux.tolist()])
        err_p = np.array([new_df.err.tolist()])
    
        point1 = np.array(feature_genarate1(time_p,flux_p,err_p))
        
        
        
        #result
        if c1_model.predict(point1)[0] == 0:
            
            point2 = np.array(feature_genarate2(time_p,flux_p,err_p))
            
            if c3_model.predict(point2)[0] == 0:
                final_reult = " This is a Binary + Pulsation star"
            elif c3_model.predict(point2)[0] == 1:
                final_reult = "This is a Pure Binary star"

        else:
            final_reult = "This is a Pulsation star"
            
        st.success(final_reult) # display result    
        plot(time_p[0],flux_p[0])
        
    
    



# Main Prediction page

if (selected == 'Pulsation stars'):
    
    # page title
    st.title('Pulsation Type Prediction')
    
    
    # layout of the page
    # make three columns in one line and place three textboxes
    # these order must be our training data set column order
    # these are page text boxes
   
    col1,col2 = st.columns(2)
    
    with col1:
        KIC_No = st.text_input('KIC Number')
        
    
    # code for Prediction 
    
    final_reult = '' # final result
    
    
    # creating a button for prediction 
    
    
    if st.button('Type of  the star'):
        
        lc  = lk.search_lightcurve( KIC_No, author = 'Kepler', cadence = 'long', quarter = 9).download()
        lc.plot()
        # extracrt dat 

        time_t = np.array(lc.time.to_value('jd'))
        time_t = time_t - time_t[0]
        flux_t = np.array(lc.flux.to_value())
        err_t = np.array(lc.flux_err.to_value())
        
        time_t = time_t.tolist()
        flux_t = flux_t.tolist()
        err_t = err_t.tolist()
        
        # create data frame
        
        data = {'time':time_t , 'flux':flux_t, 'err':err_t}
        df = pd.DataFrame(data)
        
        # fill nans using backword (main) and forward( if last point NaN) fill methods.
        
        new_df_back = df.fillna(method='bfill')
        new_df = new_df_back.fillna(method='ffill')
        
        
        # predict 
        
        time_p = np.array([new_df.time.tolist()])
        flux_p = np.array([new_df.flux.tolist()])
        err_p = np.array([new_df.err.tolist()])
    
        point3 = np.array(feature_genarate3(time_p,flux_p,err_p))
        
        
        #results
        if c2_model.predict(point3)[0] == 0:
            final_reult = "This is a Delta Scuti star"
        elif c2_model.predict(point3)[0] == 1:
            final_reult = "This is a Gamma Doradus star"
        elif c2_model.predict(point3)[0] == 2:
            final_reult = " This is a RR Lyrae star"
        elif c2_model.predict(point3)[0] == 3:
            final_reult = " This is a Solar-Like star"
        
        
        st.success(final_reult) # display result    
        plot(time_p[0],flux_p[0])
            