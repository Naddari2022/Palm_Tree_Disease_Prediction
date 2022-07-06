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
from imblearn.combine import SMOTETomek


images = "images.7z"
with py7zr.SevenZipFile('images.7z', mode='r') as z:
    z.extractall()
################################### get workin_dir , join it to images    
working_dir= os.getcwd()
train_data_dir = os.path.join(working_dir,'images/train')
test_data_dir = os.path.join(working_dir,'images/test')
test_image_file_dir = os.path.join(working_dir,'images/test/all_classes')
#############################################################
list_dir= os.path.join(test_image_file_dir,'{}')
test_imgs = [list_dir.format(i) for i in os.listdir(test_image_file_dir)]
#####################################################
#######################################################################
#  set the dimensions of our images.
img_width, img_height = 75, 75
# set the path to save our model 
top_model_weights_path = 'abs_model.h5'
# set number of the epochs and batch_size
epochs = 10
batch_size = 16
############################################
CATEGORIES=["Brown_Spot","Healthy","White_Scale"]
def create_training_data(dir):
    X=[]
    y=[]
    for category in CATEGORIES:
        path = os.path.join(dir,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(img_width,img_height))
                X.append(new_array)
                y.append(class_num)
            except Exception as e:
                pass  
    return X, y  

#####################################################################
X,Y=create_training_data(train_data_dir)
##################Prepare data shape to balanced it using SMOTE(oversampling)########################
# to be able to use the smote function to fairly distribute
# data between classes we need to make sure X is of dimension <=2
X=np.array(X).reshape(-1,img_width*img_height*3)
#Y aswell is needed as an array of dimension 1
Y=np.array(Y)
######################oversample training data ##################

smt = SMOTETomek(sampling_strategy='not majority',random_state=42)
X, Y= smt.fit_resample(X, Y)

############################display data distribution : can be plotted as a sep method
import seaborn as sns
sns.countplot(Y)
plt.title('Labels for categories: brown_spot, healthy,white_scale ')
##############Get back data shape 4 dimentional  for the feautres##########################
X = X.reshape((X.shape[0],img_width, img_height, 3))
##############Binarize the training labels ##########print("**********changing the labels to binary matrices )********************")
Y = label_binarize(Y, classes=[0 ,1 ,2])
n_classes = Y.shape[1]
################Method to Split data ####################
def Split_Data(X,Y):
	feat_train,feat_val, label_train, label_val= train_test_split(X, Y, 
                                               test_size=0.20)
	feat_train = feat_train.reshape((feat_train.shape[0], img_width,img_height, 3))
	feat_val = feat_val.reshape((feat_val.shape[0], img_width,img_height, 3))
	return feat_train,feat_val, label_train, label_val
#######################################################################
feat_train,feat_val,label_train,label_val=Split_Data(X,Y)
############Method to read and preprocess test_data##########################

def read_and_process_test_image(list_of_images):
    X = [] # images
    Y = [] # labels    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (img_width,img_height), interpolation=cv2.INTER_CUBIC)) 
        if 'brownspots' in image:
            Y.append(0)
        elif 'healthy' in image:
            Y.append(1)
        elif 'wsphase' or 'WSstage' in image:
            Y.append(2)
    
    return X, Y
################################################################################
feat_test,label_test=read_and_process_test_image(test_imgs)
label_test=np.array(label_test) 
feat_test=np.array(feat_test).reshape(-1,img_width,img_height,3)
##########binarize label_test##################
label_test = label_binarize(label_test, classes=[0 ,1 ,2])
################Method to extract features using VGG.16####################

def save_bottleneck_features(): 
    model = applications.VGG16(include_top=False, weights='imagenet',input_shape=(img_width, img_height,3))
    #build the datagenerator:
    datagen  = ImageDataGenerator(
                    rescale=1./255
                    )
    ############first generator for feat_train(including label_train)
    print("****************************************************")
    print("extracting bottle neck features training")
    train_generator = datagen.flow(
    feat_train,
    label_train,
    batch_size=batch_size,
    shuffle=False)
    nb_train_samples = len(feat_train)
    print("number of  training samples",nb_train_samples)
    ######################train####################################
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    bottleneck_features_train = model.predict_generator(
        train_generator,predict_size_train)
    np.save(open('absolute_train.npy', 'wb'),
            bottleneck_features_train)
    print("****************************************************")
    ##############valid##################################
    validation_generator=datagen.flow(
    feat_val,
    label_val,
    batch_size=batch_size,
    shuffle=False)
    nb_validation_samples = len(feat_val)
    print("number of validation samples",nb_validation_samples)
    predict_size_validation = int(
         math.ceil(nb_validation_samples / batch_size))
    bottleneck_features_validation = model.predict_generator(
        validation_generator, predict_size_validation )
    np.save(open('absolute_val.npy', 'wb'),
            bottleneck_features_validation) 
    print("****************************************************")
    test_generator = datagen.flow(
    feat_test,
    label_test,
    batch_size=batch_size,
    shuffle=False)
    nb_test_samples = len(feat_test)
    print("number of testing samples",nb_test_samples)
    predict_size_test = int(
        math.ceil(nb_test_samples / batch_size))
    bottleneck_features_test = model.predict_generator(
        test_generator, predict_size_test)
    np.save(open('absolute_test.npy', 'wb'),
            bottleneck_features_test)
    print("****************************************************")
#################################################################  
save_bottleneck_features()
#######################
def train_top_model():  
# NOTE : THE COICE OF THIS PROCESS : THE USE OF FEAT.EXTRACT
# AND DATA.GEN.FLOW() WOULDNT ALLOW ME TO USE 
# get the class lebels for the training data, in the original order  
# AS WELL AS FOR VALIDATION AND TESTING WOULD NOT ALLOW ME TO USE
# THE FOLLOWING LINE OF CODE train_labels = generator_top.classes
# THATS WHY IN THE OTHER EXAMPLE FOLLOWED I INTRODUCED  generator_top
# THATS WHY HERE I USED THE TECHNIQUE OF BINARIZING THE DATA, AND THATS 
# OF COURSE DUE TO THE USE OF CATEGORICAL CROSSENTROPY TO BE ABLE 
# TO PLOT ROC CURVES  
    train_data = np.load('absolute_train.npy')
    validation_data = np.load('absolute_val.npy')
    test_data = np.load('absolute_test.npy')
    print("****************************************************")
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, label_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, label_val))
    model.save_weights(top_model_weights_path)
    model.save('absolute.model')
    # (eval_loss, eval_accuracy) = model.evaluate(
    #     test_data,label_test, batch_size=batch_size, verbose=1)
    # print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    # print("[INFO] Loss: {}".format(eval_loss))
    # print("**************************************************")
    # y_predicted = model.predict_classes(test_data)
    # class_dictionary = {'Brown_Spot':0 ,'Healthy':1 ,'White_Scale':2}
    # print("Class dictionary representation:",class_dictionary)
    # print("Predicted  categories of leaves:",y_predicted)
    # print("Origianl categories of leaves:",label_test)
    # print("*************************************************") 
    ####################################################################
    print("*****************plot accuracy and loss***********************************")
    plt.figure(1)
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
##############################################################
train_top_model()  
###################################################################
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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
#################################################################################   
import tensorflow as tf
new_model = tf.keras.models.load_model('absolute.model')
test_data = np.load('absolute_test.npy')
y_prob = new_model.predict_proba(test_data)
pred= new_model.predict_classes(test_data)  
#########transform the onehotencoded test labels to its original format 
test_labels_oneset = np.argmax(label_test, axis = 1)
print("***************classification report & matrix**********")
matrix = confusion_matrix(test_labels_oneset,pred, labels=[0,1,2])
print('Confusion matrix : \n',matrix)
# classification report for precision, recall f1-score and accuracy
matrix_report = classification_report(test_labels_oneset,pred, labels=[0,1,2])
print('Classification report : \n',matrix_report)
###################################################################################
print("********* call the method to plot confusion matrix*************************")
plot_confusion_matrix(cm           = np.array(matrix), 
                  normalize    = True,
                  target_names = ['Brown_Spot', 'Healthy', 'White_Scale'],
                  title        = "Confusion Matrix")



 
#################################method to plot Roc FOR ALL CLASSES#############################################
def plot_roc():
    # Plot linewidth.
    lw = 2
    n_classes=3
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(label_test.ravel(), y_prob.ravel())
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

###############################################################
plot_roc()


