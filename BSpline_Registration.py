# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:20:38 2018

@author: Fakrul-IslamTUSHAR
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:54:09 2018

@author: Fakrul-IslamTUSHAR
"""

##################import Libraries##################################
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import clear_output
from glob import glob
import os
from math import pi
import time
from datetime import timedelta

#########################Import Libaries#############################
start_time = time.time()
# =============================================================================
# Function Definitions
# =============================================================================

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
  
#############################Functions Done####################################  
     
# =============================================================================
# ###########################Input_Images######################################
# =============================================================================
fixed_image =  sitk.ReadImage('pt71_CT.nii.gz', sitk.sitkInt16)
moving_image = sitk.ReadImage('pt89_CT.nii.gz', sitk.sitkInt16)
##Name the Image
Registered_imageName='Fix71_moving_89_BSc11'
#Transformation_imageName='Fix__moving_'+filename2[:5]
###Shoe the Image
interact(display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,moving_image.GetSize()[2]-1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)));
#################################################Input Done ############################################

# =============================================================================
#Registartion Start
# =============================================================================
#registration Method.
registration_method = sitk.ImageRegistrationMethod()
    
# Determine the number of BSpline control points using the physical spacing we want for the control grid. 
#############################Initializing Initial Transformation##################################
grid_physical_spacing = [100.0, 100.0, 100.0] # A control point every 50mm
image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
mesh_size = [int(image_size/grid_spacing + 0.5)\
             for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image,transformDomainMeshSize = mesh_size, order=2)
registration_method.SetInitialTransform(initial_transform)
#######################Matrix###################################################3   
registration_method.SetMetricAsMeanSquares()
#registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
##################Multi-resolution framework############3           
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
##############Interpolation#################################
registration_method.SetInterpolator(sitk.sitkLinear)
##################Optimizer############################
#registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=9, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

#######################################Print Comment#############################################
# Connect all of the observers so that we can perform plotting during registration.
registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
#################Transformation###################################################################
final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),sitk.Cast(moving_image, sitk.sitkFloat32))

# =============================================================================
# post processing Analysis
# =============================================================================
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#Visualize Expected Results
moving_resampled = sitk.Resample(moving_image,fixed_image,final_transform, sitk.sitkLinear, 0.2, moving_image.GetPixelID())
moving_resampled2 = sitk.Resample(fixed_image,moving_image,final_transform, sitk.sitkLinear, 0.2, moving_image.GetPixelID())
moving_resampled3 = sitk.Resample(fixed_image,moving_image,final_transform, sitk.sitkLinear, 0.2, fixed_image.GetPixelID())
interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_resampled));
################Saving Transformed images###################################33     
sitk.WriteImage(moving_resampled,Registered_imageName +'.nii.gz')
#sitk.WriteImage(moving_resampled2,Registered_imageName+'_two' +'.nii.gz')
#sitk.WriteImage(moving_resampled3,Registered_imageName +'_three'+'.nii.gz')


elapsed_time_secs = time.time() - start_time

msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

print(msg)    