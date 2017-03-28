#
# Training & testing model for Boxes and Dots
#
# @author Luke Munro
#
##

import sys, csv, numpy
import keras
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation

if len(sys.argv) < 4 or len(sys.argv) > 5:
	print ("enter input_file load/new epochs test (optional)")
	sys.exit(1)
_, input_file, load, epochs = sys.argv[:4]
reader = csv.reader(open(input_file, newline=''), delimiter=',')
data = []
for row in reader:
	data.append([int(x) for x in row])
x = numpy.array(data)
y = x[:,x.shape[1]-1]
x = numpy.delete(x, x.shape[1]-1, axis=1)
print (x.shape)
labels = keras.utils.to_categorical(y, num_classes=24)
train_data = x


if load == "new":
	print ("creating new model")
	model = Sequential()
#	PReLU = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #PReLU(alpha_initializer='zero', alpha_regularizer=None)
#	PReLU2 = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #PReLU(alpha_initializer='zero', alpha_regularizer=None)
	model.add(Dense(units=400, input_dim=24))
	model.add(Activation('tanh'))
	model.add(Dense(units=24))
	model.add(Activation('softmax'))
else:
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights('model.h5')
	model = loaded_model
	print("\nloaded model")

# optimizer = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.3, nesterov=True)
# optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Train the model
model.fit(train_data, labels, batch_size=32, verbose=1, validation_split=0.0, epochs=int(epochs))

if len(sys.argv) == 5 and sys.argv[4] == "test":
	test_file = "move_record3#S22E2.csv"
	reader = csv.reader(open(test_file, newline=''), delimiter=',')
	test_data = []
	for row in reader:
		test_data.append([int(x) for x in row])
	x = numpy.array(test_data)
	y = x[:,x.shape[1]-1]
	x = numpy.delete(x, x.shape[1]-1, axis=1)
	y_test = keras.utils.to_categorical(y, num_classes=24)
	x_test = x
	loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
	print ("\n" + test_file + "- stats")
	print ("loss = " + str(loss))
	print ("acc = " + str(acc))

	# Next test file
	test_file = "move_record3#03E2.csv"
	reader = csv.reader(open(test_file, newline=''), delimiter=',')
	test_data = []
	for row in reader:
		test_data.append([int(x) for x in row])
	x = numpy.array(test_data)
	y = x[:,x.shape[1]-1]
	x = numpy.delete(x, x.shape[1]-1, axis=1)
	y_test = keras.utils.to_categorical(y, num_classes=24)
	x_test = x
	loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
	print ("\n" + test_file + "- stats")
	print ("loss = " + str(loss))
	print ("acc = " + str(acc))
print (model.summary())

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print ("\nmodel saved")
