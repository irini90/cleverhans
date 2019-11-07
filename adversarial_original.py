from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import  numpy  as np
import  pandas  as pd
import functools
from  sklearn.preprocessing import  LabelEncoder , MinMaxScaler
# from  keras.models import  Sequential
# from  keras.layers  import  Dense , Dropout
from  keras.optimizers  import  RMSprop , adam
from  cleverhans.attacks  import FastGradientMethod, SaliencyMapMethod
from  cleverhans.utils_tf  import  model_train , model_eval , batch_eval, model_argmax
from  cleverhans.attacks_tf  import  jacobian_graph
from  cleverhans.utils  import  other_classes
from cleverhans.model import Model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import initializers
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport

import tensorflow  as tf
from tensorflow import keras
from  tensorflow.python.platform  import  flags
from  sklearn.multiclass import  OneVsRestClassifier
from  sklearn.tree import  DecisionTreeClassifier
from  sklearn.ensemble import  RandomForestClassifier , VotingClassifier
from  sklearn.linear_model import  LogisticRegression
from  sklearn.metrics import  accuracy_score , roc_curve , auc , f1_score
from  sklearn.preprocessing import  LabelEncoder , MinMaxScaler
from  sklearn.svm  import SVC , LinearSVC
import  matplotlib.pyplot  as plt

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation

plt.style.use('bmh')
FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 1, 'Number  of  epochs  to train  model') # was 20
flags.DEFINE_integer('batch_size', 256, 'Size of  training  batches ') # was 32
flags.DEFINE_float('learning_rate', 0.1, 'Learning  rate  for  training ')
flags.DEFINE_integer('nb_classes', 2, 'Number  of  classification  classes ')
flags.DEFINE_integer('source_samples', 10, 'Nb of test  set  examples  to  attack ')



print("\n--------------Start of  preprocessing  stage --------------\n")

training = pd.read_csv('training.csv', header=0, low_memory=False, na_values=['?'])
testing = pd.read_csv('testing.csv', header=0, low_memory=False, na_values=['?'])
all_data = pd.concat([training, testing])


# Drop following features
all_data.drop(['packet_id', 'num', 'timestamp', 'frame.time_epoch', 'frame.time_delta', 'frame.time_delta_displayed','frame.time_relative', 'tcp.options.timestamp.tsval', 'tcp.options.timestamp.tsecr', 'tcp.time_delta', 'tcp.time_relative', 'packet_type'], axis = 1, inplace=True)

# Generate One -Hot encoding

all_data_2 = pd.get_dummies(all_data, columns=['ip.version', 'ip.flags.rb', 'ip.flags.df', 'ip.flags.mf', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 
'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'dns.flags.response', 'dns.flags.opcode', 'dns.flags.truncated','dns.flags.recdesired', 
'dns.flags.z','dns.flags.checkdisable', "class_device_type", 'class_is_malicious', 'class_attack_type'])

# Separate training and test sets again - target = malicious or benign, exclude attack type from input data
target_columns = ['class_is_malicious_0', 'class_is_malicious_1']
exclude = ['class_attack_type_DoS', 'class_attack_type_MITM', 'class_attack_type_Scanning', 'class_attack_type_iot-toolkit', 'class_attack_type_No_attack']
exclude += [x for x in all_data_2.columns.values if "class_attack_type" in x]

features = list([ x for x in all_data_2.columns if not(x in target_columns) and not(x in exclude)])

y_train = all_data_2[0:training.shape[0]][target_columns].fillna(0).values
X_train = all_data_2[0:training.shape[0]][features].values

# balance dataset
mal_data_ids = np.where(y_train[:,1] == 1)[0]
ben_data_ids = np.where(y_train[:,0] == 1)[0]

keep = mal_data_ids.tolist() + ben_data_ids.tolist()

if len(mal_data_ids) > len(ben_data_ids):
    keep = np.random.choice(mal_data_ids, replace=False, size=len(ben_data_ids)).tolist() + ben_data_ids.tolist()
elif len(mal_data_ids) < len(ben_data_ids):
    keep = np.random.choice(ben_data_ids, replace=False, size=len(mal_data_ids)).tolist() + mal_data_ids.tolist()
keep = np.array(keep)

X_train = X_train[keep]
y_train = y_train[keep]

y_test = np.array(all_data_2[training.shape[0]:][target_columns])
X_test = all_data_2[training.shape[0]:][features]



print("training set {} samples: {}% benign, {}% malicious".format(len(y_train), y_train[:,0].sum()/len(y_train), y_train[:,1].sum()/len(y_train)))

# features = list(full2.columns [:-5])
# y_train = np.array(full2 [0:df.shape [0]][[ 'label_normal ', 'label_dos ', 'label_probe ', 'label_r2l ', 'label_u2r ']])
# X_train = full2 [0:df.shape [0]][ features]
# y_test = np.array(full2[df.shape [0]:][[ 'label_normal ', 'label_dos ', 'label_probe ', 'label_r2l ', 'label_u2r ']])
# X_test = full2[df.shape [0]:][ features]

# Scale  data121
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train)).astype(np.float32) # TODO test set should be scaled by train set 
X_test_scaled = np.array(scaler.transform(X_test)).astype(np.float32)
# fillna with zero
X_train_scaled = np.nan_to_num(X_train_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)


print("Training  dataset  shape", X_train_scaled.shape , y_train.shape)
print("Test  dataset  shape", X_test_scaled.shape , y_test.shape)
# print("Label  encoder y shape", y_train_l.shape , y_test_l.shape)
print("\n--------------End of  preprocessing  stage --------------\n")


print("\n--------------Start of  adversarial  sample  generation --------------\n")

# def  mlp_model():
#     """145Generate a MultiLayer  Perceptron  model146"""
#     model = Sequential()
#     model.add(Dense (256, activation='relu', input_shape = (X_train_scaled.shape [1],)))
#     model.add(Dropout (0.4))
#     model.add(Dense (256,  activation='relu'))
#     model.add(Dropout (0.4))
#     model.add(Dense(FLAGS.nb_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics =['accuracy'])
#     model.summary()
#     return model


def  mlp_model(input_shape, input_ph=None, logits=False):
    # """145Generate a MultiLayer  Perceptron  model146"""
    model = Sequential()

    layers = [ 
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.4),
        Dense(256,  activation='relu'),
        Dropout(0.4),
        Dense(FLAGS.nb_classes),
    ]
    
    for l in layers:
        model.add(l)
    
    if logits:
        logit_tensor = model(input_ph)

    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics =['accuracy'])
    model.summary()

    if logits:
        return model, logit_tensor
    return model
    
def  evaluate():
    """164Model  evaluation  function165"""
    eval_params = {'batch_size': FLAGS.batch_size}
    # with sess.as_default():
    #     feed_dict = {x: X_train_scaled}
    #     print(correct_preds.eval()
    #     print(train_preds[:10], y_train[:10])
    train_acc = model_eval(sess, x, y, predictions , X_train_scaled , y_train , args=eval_params)
    test_acc = model_eval(sess, x, y, predictions , X_test_scaled , y_test , args=eval_params)
    print('Train acc: {:.2f} Test  acc: {:.2f} '.format(train_acc, test_acc))
    
# Tensorflow  placeholder  variables
x = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train_scaled.shape[1]))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))
tf.compat.v1.set_random_seed(42)
model = mlp_model((None, X_train_scaled.shape[1]))

sess = tf.Session()
keras.backend.set_session(sess)

predictions = model(x)
init = tf.global_variables_initializer()
sess.run(init)

# Train  the  model
train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
    'verbose': 0}

# # model_train: 
# x, y = input, output placeholder
# predictions = model output predictions
# X_train_scaled = scaled input data
# y_train = output  data

# DEBUG
# for c in [RandomForestClassifier, NaiveBayes]: 
    # rf = c()
    # rf.fit(X_train_scaled, y_train)
    # train_pred = rf.predict(X_train_scaled)
    # print("RF acc trainnig set", accuracy_score(y_train, train_pred))

model_train(sess, x, y, predictions, X_train_scaled, y_train, evaluate=evaluate, args=train_params)

# Generate  adversarial  samples  for  all  test  datapoints
source_samples = X_test_scaled.shape[0]
# Jacobian -based  Saliency  Map
results = np.zeros((FLAGS.nb_classes, source_samples), dtype ='i')
perturbations = np.zeros((FLAGS.nb_classes, source_samples), dtype ='f')
grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

X_adv = np.zeros((source_samples, X_test_scaled.shape[1]))


print(type(model)) # <class 'keras.engine.sequential.Sequential'>
wrap = KerasModelWrapper(model)

jsma = SaliencyMapMethod(wrap, sess=sess)

"""
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param theta: (optional float) Perturbation introduced to modified
                  components (can be positive or negative)
    :param gamma: (optional float) Maximum percentage of perturbed features
    :param clip_min: (optional float) Minimum component value for clipping
    :param clip_max: (optional float) Maximum component value for clipping
    :param y_target: (optional) Target tensor if the attack is targeted
    """

# Other paper has theta = 1 and gamma = 0.1 
jsma_params = {'theta': 0.1, 'gamma': 0.01,
                'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}


# fgsm = FastGradientMethod(wrap, sess=sess)
# fgsm_params = {'eps': 0.3,
#                  'clip_min': 0.,
#                  'clip_max': 1.}

# for sample_ind in range (0, source_samples):
#     current_class = int(np.argmax(y_test[sample_ind]))

#     for target in [6]:
#         if current_class == 1:
#             break
        
#         adv_x , res , percent_perturb = jsma(x , predictions , grads , X_test_scaled [ sample_ind : ( sample_ind +1) ], jsma_params)

#         X_adv [ sample_ind ] = adv_x
#         results [ target , sample_ind ] = res
#         perturbations [ target , sample_ind ] = percent_perturb


# Loop over the samples we want to perturb into adversarial examples
samples_to_perturb = np.where(y_test[:,1] == 1)[0] # only malicious
nb_classes = 2 # malicious or benign 

def model_pred(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)

    print(probabilities, "************")

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)

adversarial_samples = []
samples_perturbed_idxs = []

for i, sample_ind in enumerate(samples_to_perturb):
    print('--------------------------------------')
    print('Attacking input %i/%i' % (i + 1, len(samples_to_perturb)))
    sample = X_test_scaled[sample_ind: sample_ind+1]

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = int(np.argmax(y_test[sample_ind]))
    target = 1 - current_class

    print('Generating adv. example for target class %i' % target)

    # This call runs the Jacobian-based saliency map approach
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
    adv_x = jsma.generate_np(sample, **jsma_params) # adversarial sample generated = adv_x
    adversarial_samples.append(adv_x)
    samples_perturbed_idxs.append(sample_ind)

    # Check if success was achieved
    adv_tgt = np.zeros((1, FLAGS.nb_classes)) # adversarial target = adv_tgt
    adv_tgt[:,target] = 1
    res = int(model_eval(sess, x, y, predictions, adv_x, adv_tgt, args={'batch_size': 1}))

    # Compute number of modified features
    adv_x_reshape = adv_x.reshape(-1)
    test_in_reshape = X_test_scaled[sample_ind].reshape(-1)
    nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
    percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

    # Update the arrays for later analysis
    results[target, sample_ind] = res
    perturbations[target, sample_ind] = percent_perturb

print('--------------------------------------')

malicious_targets = np.zeros((len(adversarial_samples), 2))
malicious_targets[:, 1] = 1

adversarial_samples = np.stack(adversarial_samples).squeeze()
original_samples = X_test_scaled[np.array(samples_perturbed_idxs)]

adv_acc = model_eval(sess, x, y, predictions, adversarial_samples, malicious_targets, args={'batch_size': FLAGS.batch_size})
malicious_test_acc = model_eval(sess, x, y, predictions, original_samples, malicious_targets, args={'batch_size': FLAGS.batch_size})
print("ADVERSARIAL: Accuracy of model on perturbed malicious samples: {}%".format(adv_acc*100))
print("NORMAL: Accuracy of model on unperturbed malicious samples: {}%".format(malicious_test_acc*100))

# Close TF session
sess.close()
