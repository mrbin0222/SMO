import math
import os
import random
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import requests
import tifffile as tiff
import yaml
from scipy.interpolate import RegularGridInterpolator, interp1d, interp2d
from skimage.draw import polygon
from skimage.filters import sobel
from skimage.morphology import binary_dilation


def sparse_ellipse_with_nuc(ra, rb, ang, x0, y0, sparsity, rat_nuc_cell, polarity):
    Nb=20
    C=np.array([[0,0.4470,0.7410],[0.8500,0.3250,0.0980],[0.929,0.694,0.125],[0.494,0.184,0.556],
                [0.466,0.674,0.188],[0.301,0.745,0.933],[0.635,0.078,0.184]])

    # only loop 1
    xpos = x0
    ypos = y0
    radm = ra
    radn = rb
    an = ang
    co = np.cos(an)
    si = np.sin(an)
    the = np.linspace(0, 2 * np.pi, Nb+1)

    # Create cell
    x = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
    y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos

    # Create nucleus
    x_nuc = radm * rat_nuc_cell * np.cos(the) * co - si * radn * rat_nuc_cell * np.sin(the) + xpos
    y_nuc = radm * rat_nuc_cell * np.cos(the) * si + co * radn * rat_nuc_cell * np.sin(the) + ypos
    x_nuc += np.random.rand(Nb+1) * x0 * 0.05
    y_nuc += np.random.rand(Nb+1) * y0 * 0.05

    # Add sparse noise
    x += np.random.randn(Nb+1) * sparsity / 2 * x0 / 7 + np.random.rand(Nb+1) * x0 / 10
    y += np.random.randn(Nb+1) * sparsity / 2 * y0 / 7 + np.random.rand(Nb+1) * y0 / 10

    # Interpolation
    mult = 10
    # extend x and y
    x_extended = np.tile(x, 3)
    y_extended = np.tile(y, 3)
    # create interpolation function
    interp_func_x = interp1d(np.arange(1, (Nb + 1) * 3 + 1), x_extended, kind='cubic')
    interp_func_y = interp1d(np.arange(1, (Nb + 1 ) * 3 + 1), y_extended, kind='cubic')
    # generate interpolation points
    interp_points = np.arange(1, (Nb + 1) * 3 , 1 / mult)
    # interpolate, the actual output is 1 less than the matlab code, but since the later part is the middle part, it can be ignored
    x_interpolated = interp_func_x(interp_points)
    y_interpolated = interp_func_y(interp_points)
    # take the middle part
    x = x_interpolated[(Nb + 1) * mult:2 * (Nb + 1) * mult + 1]
    y = y_interpolated[(Nb + 1) * mult:2 * (Nb + 1) * mult + 1]

    # extend x_nuc and y_nuc
    x_nuc_extended = np.tile(x_nuc, 3)
    y_nuc_extended = np.tile(y_nuc, 3)
    # create interpolation function
    interp_func_x_nuc = interp1d(np.arange(1, (Nb + 1) * 3 + 1), x_nuc_extended, kind='cubic')
    interp_func_y_nuc = interp1d(np.arange(1, (Nb + 1 ) * 3 + 1), y_nuc_extended, kind='cubic')
    # generate interpolation points
    interp_points_nuc = np.arange(1, (Nb + 1) * 3 , 1 / mult)
    # interpolate, the actual output is 1 less than the matlab code, but since the later part is the middle part, it can be ignored
    x_nuc_interpolated = interp_func_x_nuc(interp_points_nuc)
    y_nuc_interpolated = interp_func_y_nuc(interp_points_nuc)
    # take the middle part
    x_nuc = x_nuc_interpolated[(Nb + 1) * mult:2 * (Nb + 1) * mult + 1]
    y_nuc = y_nuc_interpolated[(Nb + 1) * mult:2 * (Nb + 1) * mult + 1]

    # Apply polarity to the nucleus
    x_pol_mov = co * radm * polarity / 4 + si * radn * polarity / 4
    y_pol_mov = si * radm * polarity / 4 + co * radn * polarity / 4
    if np.random.rand() <= 0.5:
        x_nuc += x_pol_mov
        y_nuc += y_pol_mov
    else:
        x_nuc -= x_pol_mov
        y_nuc -= y_pol_mov

    # Check values
    min_x = np.min(x)
    min_y = np.min(y)
    x = x - min_x + 1
    y = y - min_y + 1
    x = np.where(np.isnan(x) | np.isinf(x), 1, x)
    y = np.where(np.isnan(y) | np.isinf(y), 1, y)
    x_nuc = x_nuc - min_x + 1
    y_nuc = y_nuc - min_y + 1
    x_nuc = np.where(np.isnan(x_nuc) | np.isinf(x_nuc), 1, x_nuc)
    y_nuc = np.where(np.isnan(y_nuc) | np.isinf(y_nuc), 1, y_nuc)

    # Create cell and nucleus masks
    cell = np.zeros((math.ceil(max(y)/2)*2,math.ceil(max(x)/2)*2), dtype=bool)
    rr, cc = polygon(y, x)
    cell[rr, cc] = True

    nucleus = np.zeros((math.ceil(max(y)/2)*2,math.ceil(max(x)/2)*2), dtype=bool)
    rr_nuc, cc_nuc = polygon(y_nuc, x_nuc)
    nucleus[rr_nuc, cc_nuc] = True

    # if Cell is all zero, set Cell(1,1) to 1
    if np.all(cell == 0):
        cell[0, 0] = 1

    # if Nucleus is all zero, set Nucleus(1,1) to 1
    if np.all(nucleus == 0):
        nucleus[0, 0] = 1

    cn = cell.copy()
    # overlay the values of Nucleus onto Cell
    cn[nucleus > 0] = 1

    return cn, nucleus

def calc_ellipse(selblock_orientation, cell_size, cell_eccentricity, morph_dev, rat_nuc_cell, polarity):
    # Calculate cell morphology
    deg = (selblock_orientation + (np.random.rand() * 0.3 - 0.1)) * np.pi / 2
    x_axis = round(cell_size + np.random.rand() * cell_size * 0.5)
    y_axis = max(round(cell_size * cell_eccentricity + np.random.rand() * cell_size * 0.5), 1)
    size_x = round(cell_size + 2 + np.random.rand() * cell_size * 0.5)
    size_y = round(cell_size + 2 + np.random.rand() * cell_size * 0.5)
    cell, nucleus = sparse_ellipse_with_nuc(x_axis, y_axis, deg, size_x, size_y,
                                            morph_dev + (np.random.rand() * 0.1 - 0.05),
                                            rat_nuc_cell + (np.random.rand() * 0.1 - 0.05),
                                            polarity + (np.random.rand() * 0.1 - 0.05))
    return cell, nucleus

# generate filters
def nuc_cyt_mem_cell(size_images, pheno_cells, pheno_nuc, M, pheno_winner):

    pheno_winner += 1
    cytoplasm_mask = np.zeros(size_images)
    membrane_mask = np.zeros(size_images)
    nuclear_mask = np.zeros(size_images)
    
    thiscell = pheno_cells == pheno_winner
    thiscell_phenotype = pheno_winner  # assume only one type
    
    cytoplasm_mask[np.logical_xor(pheno_cells == pheno_winner, pheno_nuc == pheno_winner)] = thiscell_phenotype
    membrane_mask[sobel(thiscell) > 0] = thiscell_phenotype
    nuclear_mask[pheno_nuc == pheno_winner] = thiscell_phenotype

    # process background
    nuclear_mask[nuclear_mask == 0] = M[0] - 1
    cytoplasm_mask[cytoplasm_mask == 0] = M[0] - 1
    membrane_mask[membrane_mask == 0] = M[0] - 1
    
    return nuclear_mask, cytoplasm_mask, membrane_mask

def convert_phenotypes_to_marker_expression(size_images, M, marker_localization, marker_expression, nuclear_mask, cytoplasm_mask, membrane_mask):
    
    Im = np.zeros((size_images[0], size_images[1], M[1]))

    for marker in range(M[1]):
    # nuclear localization
        if marker_localization[marker] == 'Nuclear':
            nm=nuclear_mask.copy()
            # traverse each element of nm and assign value
            for i in range(nm.shape[0]):
                for j in range(nm.shape[1]):
                    index = int(nm[i, j]) - 1
                    nm[i, j] = marker_expression[index, marker]
            Im[:, :, marker] = nm

        # cytoplasmatic localization
        elif marker_localization[marker] == 'Cytoplasmatic':
            cm=cytoplasm_mask.copy()
            # traverse each element of cm and assign value
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    index = int(cm[i, j]) - 1
                    cm[i, j] = marker_expression[index, marker]
            Im[:, :, marker] = cm
        
        # CK定位
        elif marker_localization[marker] == 'CK':
            cm2=cytoplasm_mask.copy()
            # traverse each element of cm2 and assign value
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    index = int(cm2[i, j]) - 1
                    index2 = int(membrane_mask[i,j]) - 1
                    cm2[i, j] = marker_expression[index, marker] + 0.75 * marker_expression[index2, marker]
            Im[:, :, marker] = cm2

    return Im

# remote call api, implement using matlab to calculate GeneralPerlinNoise
def perlin_noise(height, width, scale_range, base_scale, ip='127.0.0.1:8080'):
    # API endpoint
    url = f'http://{ip}/perlin_noise'

    # request data
    data = {
        'x': height,
        'y': width,
        'iterations': scale_range,
        'saturation': base_scale
    }

    try:
        # send POST request
        response = requests.post(url, json=data)

        # check response status code
        if response.status_code == 200:
            # print returned result
            result = response.json()
            #print("Result:", result)
        else:
            print(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return np.array(result)

# remote call api, implement using matlab to calculate GeneralPerlinNoise
def perlin_noise_sparse(height, width, scale_range, base_scale, ip='127.0.0.1:8080'):
    # API endpoint
    url = f'http://{ip}/perlin_noise_sparse'

    # request data
    data = {
        'x': height,
        'y': width,
        'iterations': scale_range,
        'saturation': base_scale
    }

    try:
        # send POST request
        response = requests.post(url, json=data)

        # check response status code
        if response.status_code == 200:
            # print returned result
            result = response.json()
            #print("Result:", result)
        else:
            print(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return np.array(result)

def add_perlin_noise_background(Im, M, sizeImages, inte, BackgroundPerlinNoise, GT_Nb_cut, ip):
    for mk in range(M[1]):
        GeneralPerlinNoise = perlin_noise(sizeImages[0], sizeImages[1], 
                                        inte, BackgroundPerlinNoise[0],
                                        ip)
        Im[:, :, mk] = Im[:, :, mk] + GeneralPerlinNoise * 0.15 * (GT_Nb_cut > 0)
    return Im

def add_marker_perlin_noise(Im, M, size_images, marker_localization, 
                            perlinpresistence, perlinfreq, 
                            perlinpresistence_sparse, perlinfreq_sparse, ip):
    
    for mk in range(M[1]):
        if perlinpresistence_sparse[mk]>0:
            start = int(perlinfreq_sparse[mk][0])
            end = int(perlinfreq_sparse[mk][1])
            # create a list of all positive integers from the second item to the third item
            inte = list(range(start, end + 1))
            sparse_noise = perlin_noise_sparse(size_images[0], size_images[1], 
                                                inte, 
                                                perlinpresistence_sparse[mk],
                                                ip)
            Im[:, :, mk] *= sparse_noise

        start = int(perlinfreq[mk][0])
        end = int(perlinfreq[mk][1])
        # create a list of all positive integers from the second item to the third item
        inte = list(range(start, end + 1))
        general_noise = perlin_noise(size_images[0], size_images[1], 
                                            inte, 
                                            perlinpresistence[mk],
                                            ip)
        Im[:, :, mk] *= general_noise

    return Im

# remote call api, implement using matlab to calculate GeneralPerlinNoise
def imgaussfilt(Im_i, gaussfiltimage_i, ip='127.0.0.1:8080'):
    # API endpoint
    url = f'http://{ip}/imgaussfilt'

    # convert ndarray to list
    Im_list = Im_i.tolist()
    # request data
    data = {
        'Im': Im_list,
        'gaussFiltImage': gaussfiltimage_i
    }

    try:
        # send POST request
        response = requests.post(url, json=data)

        # check response status code
        if response.status_code == 200:
            # print returned result
            result = response.json()
            #print("Result:", result)
        else:
            print(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return np.array(result)

# remote call api, implement using matlab to calculate GeneralPerlinNoise
def awgn(Im_i, snr, ip='127.0.0.1:8080'):
    # API endpoint
    url = f'http://{ip}/awgn'

    # convert ndarray to list
    Im_list = Im_i.tolist()
    # request data
    data = {
        'Im': Im_list,
        'SNR': snr
    }

    try:
        # send POST request
        response = requests.post(url, json=data)

        # check response status code
        if response.status_code == 200:
            # print returned result
            result = response.json()
            #print("Result:", result)
        else:
            print(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return np.array(result)

def save_multispectral_image(Im, M, dpi, filename):
    # process and save each channel
    for chan in range(M[1]):
        # shift the minimum value of the current channel to zero
        Im[:, :, chan] -= np.min(Im[:, :, chan])
        
        # normalize
        Im[:, :, chan] /= np.max(Im)
        
        # append to save to TIFF file
        #imageio.imwrite(filename, Im[:, :, chan], format='tiff')
    Im = Im.transpose(2,0,1)
    tiff.imwrite(filename, (Im * 255).astype(np.uint8),photometric='minisblack',resolution=(dpi,dpi))
    print(f'Done: {filename_combined}, {filename_combined_z}, {filename}')
    return

def testfig():
    # show result
    fig, axes = plt.subplots(1,3,figsize=(6,6))
    axes[0].imshow(Im[:,:,0], cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Im 0')
    axes[1].imshow(Im[:,:,1], cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Im 1')
    axes[2].imshow(Im[:,:,2], cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Im 2')
    plt.show()
    return


# read YAML file
with open('python_parameters.yaml', 'r') as file:
    params = yaml.safe_load(file)

# access parameters
Selblock_Orientation_edge = params['Selblock_Orientation']
Selblock_Orientation = round(random.uniform(Selblock_Orientation_edge[0], Selblock_Orientation_edge[1]), 4)
PhenoSize = params['PhenoSize']
PhenoEccentricity = params['PhenoEccentricity']
PhenoMorphDeviation = params['PhenoMorphDeviation']
RatioNucleousCellSize = params['RatioNucleousCellSize']
PhenoPolarity = params['PhenoPolarity']
CPname = params['CPname']
Pheno_winner = params['Pheno_winner']
if Pheno_winner == -1:
    Pheno_winner = random.choice(np.arange(len(CPname)).tolist())
phname = CPname[Pheno_winner].replace(' ','')

z = params['expandsize']
dpi = params['dpi']
basefolder = params['basefolder']
ip= params['ip']

# define parameters
M = params['M']
marker_localization = params['marker_localization']
marker_expression = np.array(params['marker_expression']['array_param'])
backgroundperlinnoise = np.array(params['backgroundperlinnoise'])
perlinpresistence = np.array(params['perlinpresistence'])
perlinfreq = np.array(params['perlinfreq']['array_param'])
perlinpresistence_sparse = np.array(params['perlinpresistence_sparse'])
perlinfreq_sparse = np.array(params['perlinfreq_sparse']['array_param'])
gaussfiltimage = np.array(params['gaussfiltimage'])
SNR = np.array(params['SNR'])
gaussFiltLeakage_Left = np.array(params['gaussFiltLeakage_Left'])
gaussFiltLeakage_Right = np.array(params['gaussFiltLeakage_Right'])


# get current date and time
current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")# 生成文件名
filename_combined = f"cpimage_gt_all_{phname}_{current_time}.tiff" 
filename_combined_z = f"cpimage_gt_all_{phname}_{str(z)}x{str(z)}_{current_time}.tiff"
filename_end = f"cpimage_fluo_all_{phname}_{str(z)}x{str(z)}_{current_time}.tiff"
filename_combined = os.path.join(basefolder, filename_combined)
filename_combined_z = os.path.join(basefolder, filename_combined_z)
filename_end = os.path.join(basefolder, filename_end)

# generate cell phenotype
# call calc_ellipse function
cell, nucleus = calc_ellipse(Selblock_Orientation, PhenoSize[Pheno_winner], PhenoEccentricity[Pheno_winner], 
                             PhenoMorphDeviation[Pheno_winner], RatioNucleousCellSize[Pheno_winner], PhenoPolarity[Pheno_winner])

# process image
# assign different gray values to cell_mask and nucleus_mask, original size
combined_mask = np.zeros_like(cell, dtype=float)
combined_mask[cell > 0] = 0.5  # for example, cell_mask is light gray
combined_mask[nucleus > 0] = 0.8  # for example, nucleus_mask is dark gray

# assume the size of cell_mask and nucleus_mask is expanded to 100x100 background
a, b = cell.shape[0], cell.shape[1]

# create blank zxz image
#image = np.zeros((z, z), dtype=float)
image_cell = np.zeros((z, z), dtype=bool)
image_nus = np.zeros((z, z), dtype=bool)

# place cell_mask in the middle of the image
#image[int(z/2-a/2):int(z/2-a/2)+a, int(z/2-b/2):int(z/2-b/2)+b] = combined_mask
image_cell[int(z/2-a/2):int(z/2-a/2)+a, int(z/2-b/2):int(z/2-b/2)+b] = cell
image_nus[int(z/2-a/2):int(z/2-a/2)+a, int(z/2-b/2):int(z/2-b/2)+b] = nucleus

# create three channels images
channel_1 = np.zeros_like(cell, dtype=float)  # cell + nucleus
channel_2 = np.zeros_like(cell, dtype=float)  # cell
channel_3 = np.zeros_like(cell, dtype=float)  # nucleus
# fill channels
channel_1[cell > 0] = 0.5
channel_1[nucleus > 0] = 0.8
channel_2[cell > 0] = 1.0
channel_3[nucleus > 0] = 1.0
# merge channels
combined_image = np.stack((channel_1, channel_2, channel_3), axis=-1)
combined_image = combined_image.transpose(2,0,1)
# save as TIFF file
tiff.imwrite(filename_combined, (combined_image * 255).astype(np.uint8),photometric='minisblack',resolution=(dpi,dpi))

# create three channels images zxz
channel_1 = np.zeros((z, z), dtype=float)  # cell + nucleus
channel_2 = np.zeros((z, z), dtype=float)  # cell
channel_3 = np.zeros((z, z), dtype=float)  # nucleus
# fill channels
channel_1[image_cell > 0] = 0.5
channel_1[image_nus > 0] = 0.8
channel_2[image_cell > 0] = 1.0
channel_3[image_nus > 0] = 1.0
# merge channels
combined_image = np.stack((channel_1, channel_2, channel_3), axis=-1)
combined_image = combined_image.transpose(2,0,1)
# save as TIFF file, 3 channels, cell+nucleus, cell, nucleus
tiff.imwrite(filename_combined_z, (combined_image * 255).astype(np.uint8),photometric='minisblack',resolution=(dpi,dpi))




# parameter part
size_images=[z,z]
# assume the size of cell_mask and nucleus_mask is expanded to 100x100 background
a, b = cell.shape[0], cell.shape[1]

# create blank image with size from params
image_cell = np.zeros((z, z), dtype=float)
image_nucleus = np.zeros((z, z), dtype=float)

# place cell_mask in the middle of the image
image_cell[int(z/2-a/2):int(z/2-a/2)+a, int(z/2-b/2):int(z/2-b/2)+b] = cell
image_nucleus[int(z/2-a/2):int(z/2-a/2)+a, int(z/2-b/2):int(z/2-b/2)+b] = nucleus
pheno_cells = image_cell.copy()
pheno_nuc = image_nucleus.copy()

# align with matlab code, remember here the actual index should be - 1
pheno_cells[pheno_cells == 1.0] = Pheno_winner + 1
pheno_nuc[pheno_nuc == 1.0] = Pheno_winner + 1

# generate nuclear mask, cytoplasm mask, membrane mask
nuclear_mask, cytoplasm_mask, membrane_mask = nuc_cyt_mem_cell(size_images , pheno_cells, pheno_nuc, M, Pheno_winner)

# generate 3 channels marker expression
Im = convert_phenotypes_to_marker_expression(size_images, M, marker_localization, marker_expression, nuclear_mask, cytoplasm_mask, membrane_mask)


# get the second and third items
start = int(backgroundperlinnoise[1])
end = int(backgroundperlinnoise[2])
# create a list of all positive integers from the second item to the third item
inte = list(range(start, end + 1))

Im = add_perlin_noise_background(Im, M, size_images, inte,
                                  backgroundperlinnoise, pheno_cells,
                                  ip)

Im = add_marker_perlin_noise(Im, M, size_images, marker_localization, 
                             perlinpresistence, perlinfreq,
                             perlinpresistence_sparse,perlinfreq_sparse,
                             ip)

# Apply psf
for i in range(Im.shape[-1]):
    Im[:, :, i] = imgaussfilt(Im[:,:,i], gaussfiltimage[i], ip)

# image noise and blurring
for i in range(Im.shape[-1]):
    Im[:, :, i] = awgn(Im[:,:,i], SNR[i], ip)

for i in range(gaussFiltLeakage_Left.shape[0]-1):
    Im[:,:,i+1] = Im[:,:,i+1] + Im[:,:,i] * gaussFiltLeakage_Left[i]
for j in range(1,gaussFiltLeakage_Right.shape[0]):
    Im[:,:,j-1] = Im[:,:,j-1] + Im[:,:,j] * gaussFiltLeakage_Right[j]

save_multispectral_image(Im, M, dpi, filename_end)

