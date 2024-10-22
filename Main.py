import numpy as np
import cv2 as cv
from numpy import matlib
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os
from AOA import AOA
from BWO import BWO
from Global_Vars import Global_Vars
from Model_AR2Unet_plus_plus import Model_AR2Unet_plus_plus
from Model_CNN import Model_CNN
from Model_GRU import Model_GRU
from Model_RAN import Model_RAN
from Model_RAN_GRU import Model_RAN_GRU
from Model_ResUnet import Model_ResUnet
from Model_Resnet import Model_RESNET
from Model_UNET import Model_Unet
from Model_unet_3_plus import Model_unet_3_plus
from Obj_fun import objfun_Segmentation
from PROPOSED import PROPOSED
from Plot_results import *
from RTH import RTH
from SSA import SSA


# Read the Dataset
an = 0
if an == 1:
    Dataset_folders = './Dataset/'
    Dataset_path = os.listdir(Dataset_folders)
    Images = []
    Targets = []
    for n in range(len(Dataset_path)):
        Clases_fold = Dataset_folders + Dataset_path[n] + '/'
        Img_Fold = os.listdir(Clases_fold)
        for i in range(len(Img_Fold)):
            print(n, len(Dataset_path), i, len(Img_Fold))
            Image_Path = Clases_fold + Img_Fold[i]
            image = cv.imread(Image_Path)
            image = cv.resize(image, (256, 256))
            image = np.uint8(image)

            name = Image_Path.split('/')[-2]
            Images.append(image)
            Targets.append(name)

    label_encoder = LabelEncoder()
    Tar_encoded = label_encoder.fit_transform(Targets)
    class_tar = to_categorical(Tar_encoded, dtype="uint8")

    index = np.arange(len(Images))
    np.random.shuffle(index)
    Images = np.asarray(Images)
    Shuffled_Images = Images[index]
    Shuffled_Target = class_tar[index]

    np.save('Shuffled_Index.npy', index)
    np.save('Images.npy', Shuffled_Images)
    np.save('Targets.npy', Shuffled_Target)

# Create GT
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Ground_Truth = []
    for j in range(len(Images)):
        print(j, len(Images))
        image = Images[j]
        result_image = np.zeros(image.shape, dtype=np.uint8)
        max_val = np.max(image)
        thresh = 190
        index = np.where(image >= thresh)
        result_image[index[0], index[1]] = 255
        result_image = result_image.astype(np.uint8)
        Ground_Truth.append(result_image)
    np.save('Ground_Truth.npy', Ground_Truth)


# optimization for Segmentation
an = 0
if an == 1:
    Feat = np.load('Images.npy', allow_pickle=True)  # Load the Images
    Target = np.load('Ground_Truth.npy', allow_pickle=True)  # Load the Target
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Epoch, Step per epoch in AR2Unet++
    xmin = matlib.repmat(np.asarray([5, 5, 300]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 1000]), Npop, 1)
    fname = objfun_Segmentation
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("BWO...")
    [bestfit1, fitness1, bestsol1, time1] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("RTH...")
    [bestfit3, fitness3, bestsol3, time3] = RTH(initsol, fname, xmin, xmax, Max_iter)  # RTH

    print("SSA...")
    [bestfit4, fitness4, bestsol4, time4] = SSA(initsol, fname, xmin, xmax, Max_iter)  # SSA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Enchanced FHO

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_Seg.npy', np.asarray(BestSol))  # Bestsol classification

# Segmentation
an = 0
if an == 1:
    Data_path = './Images/Original_images/'
    Data = np.load('Images.npy', allow_pickle=True)  # Load the Data
    BestSol = np.load('BestSol_Seg.npy', allow_pickle=True) # Load the Data
    Target = np.load('Ground_Truth.npy', allow_pickle=True)  # Load the ground truth
    Unet = Model_Unet(Data_path)
    unet_3_plus = Model_unet_3_plus(Data, Target)
    ResUnet = Model_ResUnet(Data, Target)
    AR2Unet_plus_plus = Model_AR2Unet_plus_plus(Data, Target)
    Proposed = Model_AR2Unet_plus_plus(Data, Target, BestSol[4, :])
    Seg = [Unet, unet_3_plus, ResUnet, AR2Unet_plus_plus, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy', Seg)


# Classification
an = 0
if an == 1:
    Segmented = np.load('Segmented_image.npy', allow_pickle=True)[:100]
    Target = np.load('Targets.npy', allow_pickle=True)[:100]
    Feat = Segmented
    EVAL = []
    Epoch = [50, 100, 150, 200, 250, 300]
    for ACT in range(len(Epoch)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:len(Feat)][:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[:len(Feat)][learnperc:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred1 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target, EP=Epoch[ACT])
        Eval[1, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, EP=Epoch[ACT])
        Eval[2, :], pred3 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, EP=Epoch[ACT])
        Eval[3, :], pred4 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target, EP=Epoch[ACT])
        Eval[4, :], pred5 = Model_RAN_GRU(Train_Data, Train_Target, Test_Data, Test_Target, EP=Epoch[ACT])
        EVAL.append(Eval)
    np.save('Eval_all_Epoch.npy', np.asarray(EVAL))


plot_conv()
ROC_curve()
Plot_KFold()
Plot_Epoch()
plot_results_Seg()
Image_segment_comparision()
