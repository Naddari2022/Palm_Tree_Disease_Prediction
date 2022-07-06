import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation ,Flatten ,Conv2D ,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os
import cv2
import numpy as np
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import imblearn
from sklearn.model_selection import train_test_split
import py7zr
from sklearn.utils import class_weight
#######################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#######################
import itertools
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from collections import Counter
############################################################

# dimensions of our images.
img_width, img_height = 75, 75
top_model_weights_path = 'Xception_model.h5'
epochs = 10
batch_size = 16
###########################################################
##################################extract file 
images = "images.7z"
with py7zr.SevenZipFile('images.7z', mode='r') as z:
    z.extractall()
################################### get workin_dir , join it to images    
working_dir= os.getcwd()
train_data_dir = os.path.join(working_dir,'images/train')
test_data_dir = os.path.join(working_dir,'images/test')
images_dir = os.path.join(working_dir,'images/test/all_classes')
#################################################################
from tensorflow.keras.applications.xception import Xception

def save_bottleneck_features():   
    model =Xception(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))
  
    datagen  = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                    )

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='training')    
    print("extracting bottle neck features training")
    num_classes = len(generator.class_indices)
    print("length of the generator for train:",len(generator.filenames))
    print("printin class indices for train:",generator.class_indices)
    print("printin numb of classes for train:",num_classes)
    nb_train_samples = len(generator.filenames)
    print("number of train samples:",nb_train_samples)
    print("************************** counter")
    counter = Counter(generator.classes)
    print("here im printin the counter",counter)   
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    print("print number of predict_size_train",predict_size_train)
    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)   
    np.save(open('bnf_inc_train.npy', 'wb'),
            bottleneck_features_train)
    print("bottle_neck_feats  train predicted and saved")
    #EXTRACT BOTTLE NECK FEATURE FOR VALIDATION_DATA
    ###this transformation just resacling for validation:
    print("extracting bottle neck features training")
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation')
    num_classes = len(generator.class_indices)
    print("printin numb of classes for train:",num_classes)
    print("length of the generator for valid:",len(generator.filenames))
    print("printin class indices in validation:",generator.class_indices)    
    nb_validation_samples = len(generator.filenames)
    print("number of train samples:",nb_train_samples)
    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))
    print("print number of predict_size_train",predict_size_validation)
    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)
    np.save(open('bnf_inc_val.npy', 'wb'),
            bottleneck_features_validation)
    print("bottle_neck_feats  valid predicted and saved")
    ###EXTRACT BOTTLE NECK FEATURES FOR TEST_DATA
    print("extracting bottle neck features testin")
    datagen = ImageDataGenerator(
                    rescale=1./255)
    generator = datagen.flow_from_directory(
                 test_data_dir,  
                 target_size=(img_width, img_height),  
                 batch_size=batch_size,  
                 class_mode='categorical',  
                 shuffle=False) 
    generator.reset() 
    nb_test_samples = len(generator.filenames)
    print("printin numb of test samples:",nb_test_samples)
    predict_size_test = int(
        math.ceil(nb_test_samples / batch_size))
    print("printin numb of pred test_size:",predict_size_test)
    bottleneck_features_test = model.predict_generator(
        generator, predict_size_test)
    np.save(open('bnf_inc_test.npy', 'wb'),
            bottleneck_features_test)
    print("bottle_neck_feats test predicted and saved")
###############################################################
save_bottleneck_features()
####################################################
def train_top_model():  
    datagen_top =  ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                    )        
    generator_top_train = datagen_top.flow_from_directory(  
         train_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False,
         subset='training')  
   
    nb_train_samples = len(generator_top_train.filenames)  
    num_classes = len(generator_top_train.class_indices) 
    # load the bottleneck features saved earlier  
    train_data = np.load('bnf_inc_train.npy')
    #compute class weights
    print("************  class weight train***************************")
    class_weights_train = class_weight.compute_class_weight(
                'balanced',
                 np.unique(generator_top_train.classes), 
                 generator_top_train.classes)
    print("class weight:",class_weights_train)
    # get the class lebels for the training data, in the original order  
    train_labels = generator_top_train.classes 
   
    #convert the training labels to categorical vectors  
    train_labels = to_categorical(train_labels, num_classes=num_classes) 
    
    generator_top_val = datagen_top.flow_from_directory(  
         train_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical', 
         shuffle=False,
         subset='validation')  
    nb_validation_samples = len(generator_top_val.filenames)     
    validation_data = np.load('bnf_inc_val.npy')     
    validation_labels = generator_top_val.classes 
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)
                        
                         ###########build the top model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])   
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        class_weight=class_weights_train)
    ####save the model and weights
    model.save_weights(top_model_weights_path)
    ## here i can save the whole model with this 
    model.save('bnf_inc.model')
    
    ####### plot accuracy and loss
    plt.figure(1)
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show() 
###########################################################
train_top_model()
############################################################

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
###########################################################
import tensorflow as tf
new_model = tf.keras.models.load_model('bnf_inc.model')
test_data = np.load('bnf_inc_test.npy')
y_prob = new_model.predict_proba(test_data)
pred=new_model.predict_classes(test_data)
test_labels =np.array(
        [0] * 5+ [1] * 5+[2] * 5)
print("*****************")
print( "predicted probailities",y_prob)
print("predicted classes ",pred)
print("her is the format of my test_labels")
print(test_labels)
print("*****************")
print("***************classification report & matrix**********")
matrix = confusion_matrix(test_labels,pred, labels=[0,1,2])
print('Confusion matrix : \n',matrix)
# classification report for precision, recall f1-score and accuracy
matrix_report = classification_report(test_labels,pred, labels=[0,1,2])
print('Classification report : \n',matrix_report)
###################################################################################
print("*********plot confusion matrix*************************")
plot_confusion_matrix(cm           = np.array(matrix), 
                  normalize    = True,
                  target_names = ['Brown_Spot', 'Healthy', 'White_Scale'],
                  title        = "Confusion Matrix")

###############################################################  
###############################################################
test_labels_bin = label_binarize(test_labels, classes=[0 ,1 ,2])
###################################################################
def plot_roc():   
   # Plot linewidth.
    lw = 2
    n_classes=3
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
##################################################
plot_roc()
################################################