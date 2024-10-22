from itertools import cycle
import numpy as np
import warnings
import cv2 as cv
from prettytable import PrettyTable
from matplotlib import pylab
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence')
    Algorithm = ['TERMS', 'BWO-AR2Unet++', 'AOA-AR2Unet++', 'RTH-AR2Unet++', 'SSA-AR2Unet++', 'ESSA-AR2Unet++']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report of the Dataset',
          ' --------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, marker='.', markerfacecolor='red',
             markersize=12, label='BWO-AR2Unet++')
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, marker='.', markerfacecolor='green',
             markersize=12, label='AOA-AR2Unet++')
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, marker='.', markerfacecolor='cyan',
             markersize=12, label='RTH-AR2Unet++')
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, marker='.', markerfacecolor='magenta',
             markersize=12, label='SSA-AR2Unet++')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
             markersize=12, label='ESSA-AR2Unet++')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    cls = ['Resnet', 'CNN', 'RAN', 'GRU', 'RAN-GRU']
    Actual = np.load('Targets.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('ROC Curve')
    plt.savefig(path)
    plt.show()


def Plot_Epoch():
    eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']

    Graph_Term = [0, 3, 7, 12, 14, 16, 20]

    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]


        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(Graph.shape[0])

        ax.bar(X + 0.00, Graph[:, 0], color='#0165fc', edgecolor='w', width=0.15, label="Resnet")
        ax.bar(X + 0.15, Graph[:, 1], color='#ff474c', edgecolor='w', width=0.15, label="CNN")
        ax.bar(X + 0.30, Graph[:, 2], color='#be03fd', edgecolor='w', width=0.15, label="RAN")
        ax.bar(X + 0.45, Graph[:, 3], color='#21fc0d', edgecolor='w', width=0.15, label="GRU")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="RAN-GRU")
        plt.xticks(X + 0.15, ('50', '100', '150', '200', '250', '300'))
        plt.xlabel('EPOCH', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('EPOCH vs ' + Terms[Graph_Term[j]])
        path = "./Results/EPOCH_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_KFold():
    eval = np.load('Eval_all_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']

    Table_Term = [0, 13, 14, 15, 17]
    Classifier = ['TERMS', 'Resnet', 'CNN', 'RAN', 'GRU', 'RAN-GRU']

    for i in range(eval.shape[0]):
        value = eval[i, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, Table_Term])
        print('--------------------------------------------------' + str(i + 1) + ' Fold',
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            Graph = stats[i, :5, 2]
            labels = ['BWO-AR2Unet++', 'AOA-AR2Unet++', 'RTH-AR2Unet++', 'SSA-AR2Unet++', 'ESSA-AR2Unet++']
            colors = ['#f97306', '#bf77f6', '#6e750e', '#ff474c', 'k']
            ax.bar(labels, Graph, color=colors)
            plt.xticks(labels, rotation=10)
            plt.ylabel(Terms[i - 4])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Segmented Mean vs ' + Terms[i - 4])
            path = "./Results/Mean_Seg_%s_Alg.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            Graph = np.append(stats[i, 5:9, 2], stats[i, 4, 2])
            labels = ['UNET', 'Unet3+', 'ResUnet', 'AR2Unet++', 'ESSA-AR2Unet++']
            colors = ['green', 'orange', 'purple', '#ff474c', 'k']
            ax.bar(labels, Graph, color=colors)
            plt.xticks(labels, rotation=10)
            plt.ylabel(Terms[i - 4])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Segmented Mean vs ' + Terms[i - 4])
            path = "./Results/Mean_Seg_%s_Mtd.png" % ( Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Image_segment_comparision():
    Original = np.load('Images.npy', allow_pickle=True)
    Ground_truth = np.load('Ground_Truth.npy', allow_pickle=True)
    segmented = np.load('Seg_img.npy', allow_pickle=True)

    Images = [37, 42, 43, 68, 125]
    Images = np.asarray(Images)
    for i in range(Images.shape[0]):
        Orig = Original[Images[i]]
        Seg = segmented[Images[i]]
        GT = Ground_truth[Images[i]]

        for j in range(1):
            Orig_1 = Seg[j]
            Orig_2 = Seg[j + 1]
            Orig_3 = Seg[j + 2]
            Orig_4 = Seg[j + 3]
            Orig_5 = Seg[j + 4]
            plt.suptitle('Segmented Images from dataset' + str(0+1), fontsize=20)

            plt.subplot(3, 3, 1).axis('off')
            plt.imshow(GT)
            plt.title('Ground Truth', fontsize=10)

            plt.subplot(3, 3, 2).axis('off')
            plt.imshow(Orig_1)
            plt.title('UNET', fontsize=10)

            plt.subplot(3, 3, 3).axis('off')
            plt.imshow(Orig_2)
            plt.title('Unet3+', fontsize=10)

            plt.subplot(3, 3, 5).axis('off')
            plt.imshow(Orig)
            plt.title('Original', fontsize=10)

            plt.subplot(3, 3, 7).axis('off')
            plt.imshow(Orig_3)
            plt.title('ResUnet ', fontsize=10)

            plt.subplot(3, 3, 8).axis('off')
            plt.imshow(Orig_4)
            plt.title('AR2Unet++', fontsize=10)

            plt.subplot(3, 3, 9).axis('off')
            plt.imshow(Orig_5)
            plt.title('ESSA-AR2Unet++', fontsize=10)

            path = "./Results/Image_Results/Dataset_%s_image_%s.png" % (0+1, i + 1)
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Results/Image_Results/seg_' + 'Orig_image_' + str(i + 1) + '.png',
                       Orig)
            cv.imwrite('./Results/Image_Results/seg_' + 'Ground_Truth_' + str(i + 1) + '.png',
                       GT)
            cv.imwrite('./Results/Image_Results/seg_' + 'segm_Unet_' + str(i + 1) + '.png',
                       Orig_1)
            cv.imwrite('./Results/Image_Results/seg_'+ 'segm_Unet3+_' + str(i + 1) + '.png',
                       Orig_2)
            cv.imwrite(
                './Results/Image_Results/seg_' + 'segm_ResUnet_' + str(i + 1) + '.png',
                Orig_3)
            cv.imwrite(
                './Results/Image_Results/seg_' + 'segm_AR2Unet++_' + str(i + 1) + '.png',
                Orig_4)
            cv.imwrite(
                './Results/Image_Results/seg_' + 'segm_ESSA-AR2Unet++_' + str(i + 1) + '.png',
                Orig_5)


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_KFold()
    Plot_Epoch()
    plot_results_Seg()
    Image_segment_comparision()
