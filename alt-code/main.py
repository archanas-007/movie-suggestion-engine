from movielens import *
import numpy as np  #numpy module is used for making arrays

#sklearn.metrics module is used for accuracy(root mean error) score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import sys
import time
import pickle
user = []
item = []
rating = []
rating_test = []

# Load the dataset into arrays
d = Dataset()
d.load_users("G:\college\major project\dataset\\u.user", user)  #user is a list of objects of User class
d.load_items("G:\college\major project\dataset\\u.item", item)  #item is a list of objects of Item class
d.load_ratings("G:\college\major project\dataset\\u.base", rating) #rating is a list of objects of Rating class
d.load_ratings("G:\college\major project\dataset\\u.test", rating_test) #rating_test is to be used for testing purpose

n_users = len(user)
n_items = len(item)
utility = np.zeros((n_users, n_items))  #2d array containing all zeros
for r in rating:    #Load the rating in the 2d array by traversing "rating" list
    utility[r.user_id-1][r.item_id-1] = r.rating

for i in range(n_users):
    rated = np.nonzero(utility[i])  #rated is a tuple that contains indices of non-0 values
    n = len(rated[0])   #rated[0] is of type numpy array which contains indices of non 0 values
    #store the average rating for each user 
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])  
    else:   #To handle user who have not rated any movie
        user[i].avg_r = 0.
print("Utility array contains")
print (utility)
test = np.zeros((n_users, n_items)) #initialize test array by 0 
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating   #test array will store ratings to be used during testing
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
utility_clustered = []  #utility_clustered is used to store average rating of every user corresponding to each cluster
for i in range(n_users):
    average = np.zeros(19)
    tmp = []    #tmp is used to store ratings corresponding to each cluster
    for m in range(19):
        tmp.append([])
    for j in range(n_items):
        if utility[i][j] != 0:
            tmp[cluster.labels_[j] - 1].append(utility[i][j])   #labels_[j] is used to indicate to which cluster the movie j belongs
    for m in range(19):
        if len(tmp[m]) != 0:
            average[m] = np.mean(tmp[m])
        else:   #To handle cluster which is not rated by the user
            average[m] = 0
    utility_clustered.append(average)   
utility_clustered = np.array(utility_clustered)

for i in range(n_users):
    x = utility_clustered[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)  #compute average rating given by the user
    
def pcs(x, y):  #Function to compute pearson correlation similarity
    num = 0
    den1 = 0
    den2 = 0
    A = utility_clustered[x - 1]
    B = utility_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den
print
pcs_matrix = np.zeros((n_users, n_users))
for i in range(0, n_users):
    for j in range(0, n_users):
        if i!=j:
            pcs_matrix[i][j] = pcs(i + 1, j + 1)    #pcs_matrix will contain pcs between every pair of user
            sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, pcs_matrix[i][j]))

print ("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, pcs_matrix[i][j]))

print
print("PCS matrix")
print (pcs_matrix)
print

def norm(): #returns normalized rating by every user (rating of every cluster - average rating)
    normalize = np.zeros((n_users, 19))
    for i in range(n_users):
        for j in range(19):
            if utility_clustered[i][j] != 0:
                normalize[i][j] = utility_clustered[i][j] - user[i].avg_r
            else:
                normalize[i][j] = float('Inf')
    return normalize

def guess(user_id, i_id, top_n):
    similarity = [] #stores pcs between the user passed as an argument and all the other users
    for i in range(n_users):
        if i != user_id-1:
            similarity.append(pcs_matrix[user_id-1][i])
    temp = norm()
    temp = np.delete(temp, user_id-1, 0)    #remove the user itself
    #reverse sort the zip(similarity,temp) based on similarity(pcs)
    top = [x for (y,x) in sorted(zip(similarity,temp), key=lambda pair: pair[0], reverse=True)]
    s = 0
    c = 0
    for i in range(top_n):
        if top[i][i_id-1] != float('Inf'):
            s += top[i][i_id-1]
            c += 1
    g = user[user_id-1].avg_r if c == 0 else s/float(c) + user[user_id-1].avg_r
    if g < 1.0:
        return 1.0
    elif g > 5.0:
        return 5.0
    else:
        return g
    
#Guess the rating of cluster not rated by the existing users
utility_copy = np.copy(utility_clustered)
for i in range(n_users):
    for j in range(19):
        if utility_copy[i][j] == 0:
            sys.stdout.write("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
            sys.stdout.flush()
            utility_copy[i][j] = guess(i+1, j+1, 150)
print ("\rGuessing [User:Rating] = [%d:%d]" % (i, j))

pickle.dump( utility_copy, open("utility_matrix.pkl", "wb"))
# Predict ratings for u.test and find the mean squared error
y_true = []
y_pred = []
f = open('test.txt', 'w')
for i in range(n_users):
    for j in range(n_items):
        if test[i][j] > 0:
            f.write("%d, %d, %.4f\n" % (i+1, j+1, utility_copy[i][cluster.labels_[j]-1]))
            y_true.append(test[i][j])
            y_pred.append(utility_copy[i][cluster.labels_[j]-1])
f.close()

print
print ("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))
print
print
print("Heat map of PCS matrix")

#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#x, y = np.meshgrid(x, y)
plt.pcolormesh(pcs_matrix)
plt.colorbar() #need a colorbar to show the intensity scale
plt.show() #boom
