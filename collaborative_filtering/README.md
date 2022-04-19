## Collaborative filtering

#### 1. Introduction

Collaborative filtering like contented-based it's a traditional recommendation algorithm. But unlike content-based algorithm it uses similarity between users and items simultaneously to provide recommendations.

Like in the below image user 1 watched `Harry Potter` , `Shrek` and `The Dark Knight Rises` and user 3 also watched `Harry Potter` and`Shrek`. So for collaborative filtering it is very likely to recommend `The Dark Knight Rises` to user 3, since user 1 and user 3 are similar. But how are we going to represent this in a mathematical way.  We are going to create feature 2 vectors to represent users and movies, U denotes for user vector and V denotes for movie or item vector. Our goal is to learn these 2 vectors, so that the product between these 2 vectors can give us the predicted matrix A.

![cf1](D:\MachineLearning\recommendations\img\cf1.jpg)

Then our goal would be to minimize the loss between matrix A and dot product between U and V. To solve this problem we can use SGD or WALS algorithm. SGD is a more generic algorithm but it's slower compared with WALS.  WALS is a specified algorithm for matrix factorization.

![cf2](D:\MachineLearning\recommendations\img\cf2.jpg)

 And the reason why people call it weighted alternating least squares, it because only measure the positive samples it not enough and the negative samples need to take into account as well, because the matrix is a very sparse matrix with too many negative samples so we need to add a weight in the formula to control the contribution.

![cf3](D:\MachineLearning\recommendations\img\cf3.jpg)

Below image depicting the main difference between the SGD and WALS.

![cf4](D:\MachineLearning\recommendations\img\cf4.jpg)
