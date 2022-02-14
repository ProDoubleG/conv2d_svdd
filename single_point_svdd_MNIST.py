# %%
# import required modules
from tensorflow.keras import layers, Model
import tensorflow
import numpy
import os
# %%
# set variables
final_dimension = 28 # final depth of mappped dimension
normal_number = 7 # a number between 0~9, one of the mnist-label which we will consider as 'normal'   
violation_constant = 0.2 # coefficient to multiply with standard deviation to define range of 'normal'
# %%
# load mnist datasets
train_data,test_data = tensorflow.keras.datasets.mnist.load_data()
(train_x, train_y) = train_data
(test_x, test_y) = test_data

# %%
# reshape the mnist dataset to fit in conv2d inout shape
train_x  = train_x.reshape((60000, 28, 28, 1))
test_x = test_x.reshape((10000, 28, 28, 1))
# %%
# declare empty numpy list to organize mnist datas by label
label_0_train_data = numpy.empty((0,28,28,1),int)
label_1_train_data = numpy.empty((0,28,28,1),int)
label_2_train_data = numpy.empty((0,28,28,1),int)
label_3_train_data = numpy.empty((0,28,28,1),int)
label_4_train_data = numpy.empty((0,28,28,1),int)
label_5_train_data = numpy.empty((0,28,28,1),int)
label_6_train_data = numpy.empty((0,28,28,1),int)
label_7_train_data = numpy.empty((0,28,28,1),int)
label_8_train_data = numpy.empty((0,28,28,1),int)
label_9_train_data = numpy.empty((0,28,28,1),int)

label_0_test_data = numpy.empty((0,28,28,1),int)
label_1_test_data = numpy.empty((0,28,28,1),int)
label_2_test_data = numpy.empty((0,28,28,1),int)
label_3_test_data = numpy.empty((0,28,28,1),int)
label_4_test_data = numpy.empty((0,28,28,1),int)
label_5_test_data = numpy.empty((0,28,28,1),int)
label_6_test_data = numpy.empty((0,28,28,1),int)
label_7_test_data = numpy.empty((0,28,28,1),int)
label_8_test_data = numpy.empty((0,28,28,1),int)
label_9_test_data = numpy.empty((0,28,28,1),int)

train_label_list = list([
    label_0_train_data,
    label_1_train_data,
    label_2_train_data,
    label_3_train_data,
    label_4_train_data,
    label_5_train_data,
    label_6_train_data,
    label_7_train_data,
    label_8_train_data,
    label_9_train_data
    ])

test_label_list = list([
    label_0_test_data,
    label_1_test_data,
    label_2_test_data,
    label_3_test_data,
    label_4_test_data,
    label_5_test_data,
    label_6_test_data,
    label_7_test_data,
    label_8_test_data,
    label_9_test_data
    ])
# %%
# organize mnist train datasets and test datasets by label
data_process_needed = False
print("checking if data exists ...")
for label in range(10):
    if os.path.isfile(f'./data/mnist/mnist_train_label_{label}.npy')&os.path.isfile(f'./data/mnist/mnist_test_label_{label}.npy'):
        pass
    else:
        data_process_needed = True
        break

if data_process_needed:
    print("processing train data ...")
    for idx,label in enumerate(train_y):
        print(f"index : {idx}/{len(train_y)-1}",end="\r")
        train_label_list[label] = numpy.concatenate((train_label_list[label], numpy.expand_dims(train_x[idx],axis=0)), axis=0)
    print(f"\n")

    for i in range(10):
        print(f'label_{i}_data : {train_label_list[i].shape}')

    print("processing test data ...")
    for idx,label in enumerate(test_y):
        print(f"index : {idx}/{len(test_y)-1}",end="\r")
        test_label_list[label] = numpy.concatenate((test_label_list[label], numpy.expand_dims(test_x[idx],axis=0)), axis=0)
    print(f"\n")

    for i in range(10):
        print(f'label_{i}_data : {test_label_list[i].shape}')

    print('saving processed data ...')
    for label in range(10):
        with open(f'./data/mnist/mnist_train_label_{label}.npy', 'wb') as f:
            numpy.save(f,train_label_list[label])
        with open(f'./data/mnist/mnist_test_label_{label}.npy', 'wb') as f:
            numpy.save(f,test_label_list[label])
else:
    print('data_already_exists ...')
    for label in range(10):
        with open(f'./data/mnist/mnist_train_label_{label}.npy', 'rb') as f:
            train_label_list[label]= numpy.load(f)
        with open(f'./data/mnist/mnist_test_label_{label}.npy', 'rb') as f:
            test_label_list[label] = numpy.load(f)
# %%
# normalize the pixel data 
for label in range(10):
    test_label_list[label] = test_label_list[label]/255.0
    train_label_list[label] = train_label_list[label]/255.0
# %%
# define normal dataset whose label matches the predefined 'normal_number' will become normal_data itself
# all the other datasets whose label does not match the normal_number will be concatenated into abnormal_data
normal_data = numpy.empty((0,28,28,1),int)
abnormal_data = numpy.empty((0,28,28,1),int)

for label in range(10):
    if label == normal_number:
        normal_data = train_label_list[label]
    else:
        abnormal_data = numpy.append(abnormal_data,train_label_list[label],axis=0)
# %%
# configure SVDD which maps a input to a certain point in the defined-dimension
inputs = layers.Input(shape=(28,28,1), name="input")
layer1 = layers.Conv2D(16, (2, 2),padding='same', use_bias=False)(inputs)
pooling1 = layers.MaxPooling2D((2, 2),padding='valid')(layer1)
layer2 = layers.Conv2D(32, (2, 2),padding='same', use_bias=False)(pooling1)
pooling2 = layers.MaxPooling2D((2, 2),padding='valid')(layer2)
layer3 = layers.Conv2D(64, (2, 2),padding='same',use_bias=False)(pooling2)
pooling3 = layers.MaxPooling2D((2, 2),padding='valid',name="encoder_output")(layer3)
flat_layer =  layers.Flatten()(pooling3)
output_layer = layers.Dense(final_dimension, activation=None, use_bias=False)(flat_layer)
svdd_filter = Model(inputs,output_layer)
svdd_filter.summary()
# %%
# Basically, svdd trains ONLY with normal data and tries to map them to center of origin
# loss is defined by the norm(size of the vector) of the mapped point
def vector_norm_loss(y_true, y_pred):
    power = tensorflow.constant([2.0])
    ones = tensorflow.ones([final_dimension,], dtype=tensorflow.float32)
    power = tensorflow.math.multiply(ones, power)
    tmp = tensorflow.pow(y_pred, power)
    tmp = tensorflow.math.reduce_sum(tmp)
    vector_size = tensorflow.sqrt(tmp)
    return vector_size

adam = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
svdd_filter.compile(loss=vector_norm_loss,optimizer=adam)
callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, verbose=1)

svdd_filter.fit(
        normal_data,
        numpy.zeros((len(normal_data),final_dimension)),
        epochs=1000,
        validation_split=0.2,
        callbacks=[callback]
        )
# %%
# statistics of result of data used in training(normal data)
normal = svdd_filter.predict(normal_data)
train_vector_norm_list = []
for vector in normal :
    vector_norm = 0
    for i in vector:
        vector_norm += i*i
    vector_norm = numpy.sqrt(vector_norm)
    train_vector_norm_list.append(vector_norm)

print('normal norm mean : ', numpy.mean(train_vector_norm_list))
print('normal norm std : ', numpy.std(train_vector_norm_list))
# %%
# statistics of result of data not used in training (abnormal data)
abnormal = svdd_filter.predict(abnormal_data)
validation_vector_norm_list = []
for vector in abnormal:
    vector_norm = 0 
    for i in vector:
        vector_norm += i*i
    vector_norm = numpy.sqrt(vector_norm)
    validation_vector_norm_list.append(vector_norm)

print('abnormal norm mean : ', numpy.mean(validation_vector_norm_list))
print('abnormal norm std : ', numpy.std(validation_vector_norm_list))
# %%
# statistics of result of test data seperated by label
for label in range(10):
    test = svdd_filter.predict(test_label_list[label])
    test_vector_norm_list = []
    for vector in test:
        vector_norm = 0 
        for i in vector:
            vector_norm += i*i
        vector_norm = numpy.sqrt(vector_norm)
        test_vector_norm_list.append(vector_norm)
    
    print("test label : ",label)
    print('test norm mean : ', numpy.mean(test_vector_norm_list))
    print('test norm std : ', numpy.std(test_vector_norm_list))
# %%
# visualize the result
import matplotlib.pyplot as plt
if final_dimension ==2:
    for label in range(10):
        test = test_label_list[label]
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"normal_label : {normal_number}, test_label : {label}",fontsize=20)

        labels = ['abnormal', 'normal', 'test']
        colors=["#000000", "#00FF00", "#0000FF"]
        data_list = [abnormal,normal,test]
        alphas = [0.8,0.05,0.4]

        for label, color,data,integrity in zip(labels,colors,data_list,alphas):
            ax.scatter(data[:,0],
                        data[:,1],
                    c = color,
                    s = 10,
                    alpha=integrity)
        
        ax.legend(labels)
else:
    print('cannot plot in 2D : final_dimension != 2, plotting by 2D pca')
    from sklearn.decomposition import PCA
    for label in range(10):
        test = test_label_list[label]
        test = svdd_filter.predict(test_label_list[label])
        pca = PCA(n_components=2)
        normal_pca = pca.fit_transform(normal)
        abnormal_pca = pca.fit_transform(abnormal)
        test_pca = pca.fit_transform(test)
        pca_list=[abnormal_pca,normal_pca,test_pca]

        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title(f"normal_label : {normal_number}, test_label : {label}",fontsize=20)

        labels = ['abnormal','normal', 'test']
        colors=["#000000", "#00FF00", "#0000FF"]
        alphas = [0.8,0.05,0.4]

        for label, color,pca,integrity in zip(labels,colors,pca_list,alphas):
            ax.scatter(pca[:,0],
                        pca[:,1],
                    c = color,
                    s = 10,
                    alpha=integrity)
        ax.legend(labels)
# %%
# test with test data to check model performance
for label in range(10):
    normal_cnt = 0
    test = svdd_filter.predict(test_label_list[label])
    test_vector_norm_list = []
    for vector in test:
        vector_norm = 0 
        for i in vector:
            vector_norm += i*i
        vector_norm = numpy.sqrt(vector_norm)
        test_vector_norm_list.append(vector_norm)
    for norm in test_vector_norm_list:
        if norm <= (numpy.mean(train_vector_norm_list) + violation_constant*numpy.std(train_vector_norm_list)):
            normal_cnt += 1
        else:
            pass
    print(f'for label :{label}, {normal_cnt}/{len(test_vector_norm_list)}, {int(100*float(normal_cnt/len(test_vector_norm_list)))}% were considered as normal')
# %%
