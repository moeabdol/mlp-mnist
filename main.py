import numpy
import theano
import theano.tensor as T
from Utils import *
from Logistic_Regression import *
from MLP import *
import matplotlib.pyplot as plt
from pylab import *

learning_rate = 0.01
L1_reg = 0.00
L2_reg = 0.0001
n_epochs = 1000
dataset = 'mnist.pkl.gz'
batch_size = 20
n_hidden = 500

datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

###############
# build model #
###############
print '... building the model'
index = T.lscalar()	# index to minibatch
x = T.matrix('x')	# data of rasterized images
y = T.ivector('y')	# labels are 1D vector of int
rng = numpy.random.RandomState(1234)
classifier = MLP(rng=rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10) # MLP with Logistic regression classifier
# cost to minimize during training
cost = classifier.negative_log_likelihood(y) \
	+ L1_reg * classifier.L1 \
	+ L2_reg * classifier.L2_sqr

test_model = theano.function(inputs=[index],
		outputs=classifier.errors(y),
		givens={x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]})

valid_model = theano.function(inputs=[index],
		outputs=classifier.errors(y),
		givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

# model to show misclassified examples
misclassified_model = theano.function(inputs=[index],
		outputs=classifier.logRegressionLayer.y_pred,
		givens={x: test_set_x[index: (index + 1)]})

# compute the gradient of cost with respect to theta
gparams = []
for param in classifier.params:
	gparam = T.grad(cost, param)
	gparams.append(gparam)

# specifiy how to update the parameters of the model
updates = []
for param, gparam in zip(classifier.params, gparams):
	updates.append((param, param - learning_rate * gparam))

train_model = theano.function(inputs=[index],
		outputs=cost,
		updates=updates,
		givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]})

###############
# train model #
###############
def train():
	print '... training the model'
	epoch = 0
	while(epoch < n_epochs):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
		print 'training epoch %i' %epoch

def test():
	for minibatch_index in xrange(n_test_batches):
		minibatch_mean_error = test_model(minibatch_index)
		print 'testing minibatch %i and mean error = %f' %(minibatch_index, minibatch_mean_error)

def show_misclassified():
	for i in xrange(n_test_batches * batch_size):
		image = test_set_x[i].eval()
		label = test_set_y[i].eval()
		prediction = misclassified_model(i)[0]
		if prediction != label:
			print 'misclassified example found in test set at index %i' %i
			# show actual image
			image = image.reshape(28, 28)
			plt.imshow(image, cmap=cm.gray)
			plt.xlabel('Actual ' + str(label))
			plt.ylabel('Prediction ' + str(prediction))
			plt.show()










