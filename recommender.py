import random,time,string,numpy as np,nltk
from sklearn import metrics
from sklearn import svm
import scipy.io as sio
import re
from nltk import PorterStemmer
import matplotlib.pyplot as plt
import scipy.optimize as op
import gc
import pandas as pd
import operator

# np.set_printoptions(threshold=np.nan)
gc.collect()
gc.set_debug(gc.DEBUG_STATS)
count_iter = 1

def find_indices(ratings):
	indices = []
	for index,rating in enumerate(ratings):
		if rating==1:
			indices.append(index)
	return indices

def normalize_ratings(Y,R):
	# Y has ratings of each user for every movie
	# user-column and movie-row
	# find mean of ratings for each movie
	print 'In norm ratings'
	# raw_input()
	print 'Collecting garbage'
	gc.collect()
	print 'Garbage coll done!'
	m,n = len(Y),len(Y[0])
	Ymean = np.zeros((m,1),dtype=np.float32)
	Ynorm = np.zeros((m,n),dtype=np.float32)
	for i in xrange(m):
		# find out no of 1s in R -- no of ratings for i movie
		print 'R[i]: ',R[i][:30]
		# raw_input()
		indices = find_indices(R[i])
		print 'indices: ',indices[:30]
		# raw_input()
		num_rated = len(indices)
		print num_rated,Y[i].sum()
		# raw_input()
		Ymean[i] = float(Y[i].sum())/float(num_rated)
		print 'i Ymean: ',i,Ymean[i]
		for ind in indices:
			# print "Index: 	",ind
			# print 'Y[i][ind]: ',Y[i][ind],' Ymean[i]: ',Ymean[i]
			# raw_input()
			Ynorm[i][ind] = Y[i][ind] - Ymean[i]
			# print 'Ynorm[i,ind]: ',Ynorm[i,ind]
			# raw_input()
		print 'Original Y: ',Y[i][:10]
		print 'Mean Normalized Y: ',Ynorm[i][:10]	

		# saving Ymean and Ynorm
		np.savetxt("ymean.txt",Ymean) 
		np.savetxt("ynorm.txt",Ynorm) 

		print 'Collecting garbage'
		gc.collect()
		print 'Garbage coll done!'
	
	return Ynorm,Ymean

def cost(params,Y,R,num_users,num_features,num_movies,lamb):
	global count_iter
	print 'Iter: ',count_iter
	count_iter += 1
	print 'In Cost!'
	# raw_input()
	J = 0
	# print len(Y),len(Y[0]),len(R),len(R[0]) # 1682 X 943
	X = np.asarray(params[:num_movies*num_features])
	theta = np.asarray(params[num_movies*num_features:])
	X = np.reshape(X,(num_movies,num_features)) # 943 X 10
	theta = np.reshape(theta,(num_users,num_features)) # 1682 X 10

	# X_grad = np.zeros((len(X),len(X[0])))
	# theta_grad = np.zeros((len(theta),len(theta[0])))
	J = 0.5 * np.sum(np.sum(np.square(np.multiply(R,np.dot(X,theta.T)-Y)))) + (lamb/2) * (np.sum(np.sum(np.square(theta))) + np.sum(np.sum(np.square(X))))
	X,theta = None,None
	del(X)
	del(theta)
	print 'cost: ',J
	print 'Collecting garbage'
	gc.collect()
	print 'Garbage coll done!'
	# raw_input()
	return J

def cal_grad(params,Y,R,num_users,num_features,num_movies,lamb):
	print 'In cal grad!'
	# raw_input()
	X = np.asarray(params[:num_movies*num_features])
	theta = np.asarray(params[num_movies*num_features:])
	X = np.reshape(X,(num_movies,num_features)) # 943 X 10
	theta = np.reshape(theta,(num_users,num_features)) # 1682 X 10

	X_grad = np.zeros((len(X),len(X[0])))
	theta_grad = np.zeros((len(theta),len(theta[0])))
	
	X_grad = np.dot(np.multiply(R,np.dot(X,theta.T)-Y),theta) + lamb*X
	theta_grad = np.dot(np.multiply(R.T,np.dot(theta,X.T)-Y.T),X) + lamb*theta;
	np.savetxt("xgrad",X_grad)
	np.savetxt("thetagrad",theta_grad)
	a,b = np.hstack(X_grad),np.hstack(theta_grad)
	grad = np.array(np.concatenate((a,b)))
	X_grad,theta_grad,a,b = None,None,None,None
	del(X_grad)
	del(theta_grad)
	del(a)
	del(b)
	print len(params),len(grad)
	# raw_input()
	print 'Collecting garbage'
	gc.collect()
	print 'Garbage coll done!'
	return grad

#load spam_train.mat
f = sio.loadmat("ex8_movies.mat")
# print Y[0]
# raw_input()
# fig = plt.xlabel("Users")
# fig = plt.ylabel("Movies")
# for i in xrange(len(Y)):
# 	fig = plt.scatter(np.arange(943),Y[i])
# plt.savefig("ratings")

# randomly initialize theta and x to small values -- already done in ex8_movieParams
params = sio.loadmat("ex8_movieParams.mat")
num_users = 4#params['num_users'][0][0]-443 # 943
num_features =3# params['num_features'][0][0] # 10
num_movies = 5#params['num_movies'][0][0]-1182 # 1682
X = np.asarray(params['X'][:num_movies,:num_features]) # 1682 X 10 -- num_users X num_features
theta = np.asarray(params['Theta'][:num_users,:num_features]) # 943 X 10 -- num_movies X num_features
Y = np.asarray(f['Y'][:num_movies,:num_users])
R = np.asarray(f['R'][:num_movies,:num_users])

# print len(X),len(X[0]),len(theta),len(theta[0]),num_users,num_movies,num_features,len(Y),len(Y[0])
# raw_input()

# J,grad = cost(x0,Y,R,num_users,num_features,num_movies,1.5) 
# print 'J with reg 1.5: ',J
# raw_input()

#my_ratings
movie_file = open("movie_ids.txt","r")
movie_idx = 1
movie_list = {}
for line in movie_file.readlines():
	movie_name = filter(None,re.split("^[0-9]+ ",line))
	movie_list[movie_idx] = movie_name
	movie_idx += 1

#fmincg
num_users = params['num_users'][0][0]-643 # 943
num_features =params['num_features'][0][0] # 10
num_movies = params['num_movies'][0][0]-1382 # 1682


my_ratings = np.zeros((num_movies,1))
my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7]  = 3
my_ratings[12] = 5
my_ratings[54] = 4
my_ratings[64] = 5
my_ratings[66] = 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
# my_ratings[355] = 5

# R values for new ratings
my_ratings_idx = np.zeros((num_movies,1))
for (idd,val) in enumerate(my_ratings):
	if val:
		my_ratings_idx[idd] = 1

X = np.random.rand(num_movies,num_features)
theta = np.random.rand(num_users,num_features)
a,b = np.hstack(X),np.hstack(theta)
x0 = np.array(np.concatenate((a,b)))
x0 = np.reshape(x0,(num_movies*num_features+num_users*num_features,1))
lamb = 10	

#add my_ratings to Y and R
Y = np.insert(Y,0,values=my_ratings,axis=1)
R = np.insert(R,0,values=my_ratings_idx,axis=1)

Y = np.asarray(f['Y'][:num_movies,:num_users])
R = np.asarray(f['R'][:num_movies,:num_users])


Ynorm,Ymean = normalize_ratings(Y,R)
# Ynorm = np.loadtxt("ynorm.txt")
# Ymean = np.loadtxt("ymean.txt")

print len(X),len(X[0]),len(theta),len(theta[0]),num_users,num_movies,num_features,len(Y),len(Y[0])

options = {'disp':True,'maxiter':100}
theta = op.minimize(cost,x0,method='cg',jac=cal_grad,args=(Y,R,num_users,num_features,num_movies,lamb),options=options)
print 'Collecting garbage'
gc.collect()
print 'FINAL Garbage coll done!'
theta = np.loadtxt("thetagrad")
X = np.loadtxt("xgrad")
p = np.dot(X,theta.T)
my_predictions = {}
for i in xrange(num_movies):
	my_predictions[i+1] = p[0][i] + Ymean[i]
# ith movie idx --> prediction
# sort dictionary by key values
my_predictions = sorted(my_predictions.items(), key=operator.itemgetter(1),reverse=True)
# print my predictions along with the movie name
for (movie_id,movie_rating) in my_predictions:
	print 'Predicted rating for ',movie_list[movie_id][0].strip(),' is ',movie_rating[0]
	raw_input()