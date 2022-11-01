from movielens import *
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
import sys
import time

user = []
item = []

# Load the dataset into arrays
d = Dataset()
d.load_users("G:\college\major project\dataset\\u.user", user)
d.load_items("G:\college\major project\dataset\\u.item", item)

n = 5   #display 5 top recommended genres
n_users = len(user)
n_items = len(item)

utility_matrix = pickle.load( open("utility_matrix.pkl", "rb") )    #load the classifier

#compute the average rating for each user and stores it in the user's object
for i in range(n_users):
    x = utility_matrix[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

def pcs(x, y, ut):  #Function to compute pearson correlation similarity
    num = 0
    den1 = 0
    den2 = 0
    A = ut[x - 1]
    B = ut[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den

movie_genre = []
for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

movie_genre = np.array(movie_genre)
#19 clusters are formed because there are 19 genres 
#These clusters are randomly selected 10 times
#Maximum number of iterations for a single run are 300
cluster = KMeans(n_clusters=19, n_init = 10, max_iter = 300)

#Compute cluster centers and predict cluster index for each sample
cluster.fit_predict(movie_genre)

ask = random.sample(item, 10)   #select 10 random movies
recommended_movies = random.sample(item, 5)
new_user = np.zeros(19) #To store rating of each cluster by the new user 

user.append(User(944,20,"M","Student",110019))

print("Movie suggestion engine")
print ("Please rate the following movies (1-5):")
for movie in ask:
	print (movie.title + ": ")
	a = float(input())
	if new_user[cluster.labels_[movie.id - 1]] != 0:
	    new_user[cluster.labels_[movie.id - 1]] = (new_user[cluster.labels_[movie.id - 1]] + a) / 2
	else:
	    new_user[cluster.labels_[movie.id - 1]] = a

utility_new = np.vstack((utility_matrix, new_user)) #add new user to the already existing matrix

pcs_matrix = np.zeros(n_users)

print()
print ("Finding users which have similar preferences.(age, id, gender, occupation)")
for i in range(n_users):
    pcs_matrix[i] = pcs(944, i + 1, utility_new)    #Find pcs of 944th(new) user with the rest of the users 

user_index = []
for i in range(len(user)):
	user_index.append(i)
	i+=1
user_index = np.array(user_index)

top_n = [x for (y,x) in sorted(zip(pcs_matrix, user_index), key=lambda pair: pair[0], reverse=True)]
top_n = top_n[:n]

top_n_genre = []

for i in range(n):
	print (user[int(top_n[i])].id, user[int(top_n[i])].age,user[int(top_n[i])].sex,user[int(top_n[i])].occupation)
	print()
	maxi = 0
	maxe = 0
	for j in range(19):
		if maxe < utility_matrix[top_n[i]][j]:
			maxe = utility_matrix[top_n[i]][j]
			maxi = j
	top_n_genre.append(maxi)
	
print ("Movies you might like:")
for movie in recommended_movies:
    print(movie.title)

top_n_genre = np.unique(top_n_genre)
print()
print ("Genres you might like:")
for i in top_n_genre:
	if i == 0:
		print ("unknown")
	elif i == 1:
		print ("action")
	elif i == 2:
		print ("adventure")
	elif i == 3:
		print ("animation")
	elif i == 4:
		print ("childrens")
	elif i == 5:
		print ("comedy")
	elif i == 6:
		print ("crime")
	elif i == 7:
		print ("documentary")
	elif i == 8:
		print ("drama")
	elif i == 9:
		print ("fantasy")
	elif i == 10:
		print ("film_noir")
	elif i == 11:
		print ("horror")
	elif i == 12:
		print ("musical")
	elif i == 13:
		print ("mystery")
	elif i == 14:
		print ("romance")
	elif i == 15:
		print ("science fiction")
	elif i == 16:
		print ("thriller")
	elif i == 17:
		print ("war")
	else:
		print ("western")
		
