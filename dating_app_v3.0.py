import numpy as np
import random

## This allows the user to customize how many human qualities they want to look at.
## This is for the sake of user-friendliness rather than simulating a real dating app.
user_input = input("How many human qualities would you like to look at?")
while not(user_input.isdigit()):
    user_input = input("Please enter a whole number value:")
if user_input.isdigit():
    while int(user_input) > 10 or int(user_input) <= 0:
        user_input = input("Please enter a value less than or equal to 10 and greater than 0:")
number_qualities = int(user_input)

## Print the human qualities that the user will be looking at. 
## These are arbitrary for the calculations.
list_of_qualities = ["Humor","Empathy","Intelligence","Extroversion","Athleticism","Honesty","Confidence","Aggressiveness","Selfishness","Wealth"]
text = "The human qualities that will be represented by each vector are: "
for word in list_of_qualities[:(number_qualities-1)]:
    text += word
    text += ", "
text += list_of_qualities[number_qualities-1]
print(text)

## Generate a large set of real n vectors, and each vector represents a human.
## Note that this is a list of vectors, not a matrix.
list_of_humans = []
for i in range(500):
    human = []
    for j in range(number_qualities):
        ## Each quality is measuerd on a Likert scale from 1 to 5, with 5 indicating a lot
        ## of that quality and 1 indicating very little of that quality.
        human.append(random.randint(1,5)) ## This adds a random score for each quality for each human.
    list_of_humans.append(human)

## Make a copy of the list of humans as to not interfere with the original list.
list_copy = list_of_humans.copy()

## Make matrices to store the human vectors in, based on user input.
M_like = []
M_dislike = []

## These lists allow the user to type "yes" and "no" however they want.
yes_answers = ["yes","YES","Yes","y"]
no_answers = ["no","NO","No","n"]

## Initially, generate random vectors for the user to vote on. 
## We will gather data to make recommendations.
while len(M_dislike) < 5 or len(M_like) < 5:
    for candidate_trial_0 in range(5):
        if not list_copy: ## Although the list of humans should not be empty, this code is here to catch it just in case.
            break
        random_human = list_copy[random.randint(0,(len(list_copy)-1))] ## Generate a random vector from the list and show it to the user.
        print(random_human)
        user_input = input("Do you like them? Yes or No:")
        if user_input in yes_answers:
            M_like.append(random_human) ## If the user likes a human/vector, add them to the "like" matrix.
            list_copy.remove(random_human) ## After each human is shown to the user, it is removed. This should ensure that it is not recommended again.
        elif user_input in no_answers:
            M_dislike.append(random_human)
            list_copy.remove(random_human)
        else:
            user_input = input("Respond with Yes or No:")

## The idea is that for qualities the user does not care about, they should be random, 
## and the qualities the user does care about should follow a pattern.

## The technique we will use in this version is finding a least-squares linear regression model for the data.
## The line will be in the form y = AX, where A is the coefficients matrix and X is the matrix with the measured qualities.
## Another way to represent the line is y = c0 + c1x1 + c2x2 + ... + cnxn where each xi is a human quality and each ci is a constant.
## For our target y, we will input 1 for "liked" vectors and 0 for "disliked" vectors.

def mean_vector(matrix):
    mean_vect = np.mean(matrix, axis = 0)
    return mean_vect

## The advantage of this approach, compared to the approach in Version 1.0,
## is that this maximizes the gap between the two groups.
def lin_reg():
    X = np.array(M_like + M_dislike) ## Generate a matrix that combines the data from both matrices.
    y = np.array([1] * len(M_like) + [0] * len(M_dislike)) ## Our target will be a matrix where 1's represent "like" vectors and 0's represent "dislike" vectors.
    X_with_intercept = np.column_stack([np.ones(len(X)), X]) ## Add a column of ones to the matrix that contains all the data points.
    ## Below is the code for computing the matrix of the coefficients for the least-squares regression line using the linear algebraic formula.
    XTX = np.matmul(X_with_intercept.T, X_with_intercept)
    XTy = np.matmul(X_with_intercept.T, y)
    XTX_inv = np.linalg.inv(XTX)
    coefficients = np.matmul(XTX_inv, XTy)
    line = coefficients[1:] ## Extract the coefficient for each quality. The first item in the list is the intercept, which is not important because the line is defined by its direction.
    line = line / np.linalg.norm(line) ## Since we found the direction of the line, we now normalize the line.
    return line

## Calculate the distance between the vector the program is to recommend and the
## line for like and dislike vectors (which we found using principle eigenvectors). 
## Then calculate a "score" based on those distances.

def orth_distance(vector, mean, line):
    vector = np.array(vector)
    line = np.array(line)
    projection = mean + np.dot(vector - mean, line) * line ## This uses the formula for projection.
    orth = vector - projection
    orth_dist = np.linalg.norm(orth) ## Find the magnitude of the orthogonal component to find the orthogonal distance.
    return orth_dist

mean_vector_like = np.array(mean_vector(M_like))
mean_vector_dislike = np.array(mean_vector(M_dislike))

def calculate_score(recommended):
    line = lin_reg()
    coord = np.dot(recommended, line) ## This uses the projection formula (without multiplying by the unit vector) to get a 1D coordinate along the line.
    coord_like_clusters =  [np.dot(h, line) for h in M_like] if M_like else [coord] ## Gather the coordinates for the "liked" and "disliked" vectors.
    coord_dislike_clusters = [np.dot(h, line) for h in M_dislike] if M_dislike else [coord]
    mean_like_coord = np.mean(coord_like_clusters) ## This gives you the average coordinates for the "liked" vectors.
    mean_dislike_coord = np.mean(coord_dislike_clusters)
    distance_like = abs(coord - mean_like_coord) ## Since we used dot products earlier to get 1D coordinates, we can easily compare these 1D coordinates.
    distance_dislike = abs(coord - mean_dislike_coord)
    if distance_dislike == 0:
        score = 100 ## This checks for the case where the denominator for score is 0. Make the score some arbitrarily large number bigger than the threshold.
    else:
        score = distance_like / distance_dislike
    ## A score less than 1 indicates that the recommended vector is closer to 
    ## the line for "like" than to the line for "dislike," which is what we want in a recommendation.
    return score

## If the human fits a certain criteria, then recommend them to the user.
## Here, the threshold is set to 0.7 in an attempt not to be too restricting or lenient,
## but it can be adjusted.
threshold = 0.7

shown_humans = []

## This function will be helpful in the case that the program runs out of recommendations to give the user.
## Later in this program, we will see that this function will give the user more random vectors to gather more data.
def collect_more_data(num_samples=5):
    global M_like, M_dislike, mean_vector_like, mean_vector_dislike, shown_humans
    available = [h for h in list_of_humans if h not in shown_humans]
    if not available:
        print("No more available humans to show!")
        return False
    samples_to_show = min(num_samples, len(available))
    for i in range(samples_to_show):
        if not available:
            break
        random_human = available[random.randint(0,(len(list_copy)-1))]
        available.remove(random_human)
        shown_humans.append(random_human)
        print(random_human)
        user_input = input("Do you like them? Yes or No:")
        if user_input in yes_answers:
            M_like.append(random_human)
        elif user_input in no_answers:
            M_dislike.append(random_human)
        else:
            user_input = input("Respond with Yes or No:")
    return True

def generate_recs(matrix):
    global shown_humans, threshold
    recommendations = []
    available_humans = [h for h in matrix if h not in shown_humans] ## This is to ensure vectors already recommended are not recommended again.
    scored_humans = []
    for human in available_humans: ## Calculate scores for all remaining humans.
        score = calculate_score(human)
        scored_humans.append((human,score))
    scored_humans.sort(key=lambda x: x[1]) ## Sort by score (lowest first, i.e. most liked).
    for human, score in scored_humans[:5]: ## Look at a few humans at a time so that the user is not stuck in a loop forever.
        if score <= threshold:
            recommendations.append(human)
            shown_humans.append(human)
    return recommendations
## Generate a list of vectors suitable to initially recommend to the user.
recommendations = generate_recs(list_of_humans)

## Give recommendations, built from data collected from the user's responses.
def give_recs(matrix):
    global M_like, M_dislike, mean_vector_like, mean_vector_dislike
    if not matrix: ## This is in the case that there are no recommendations. The program keeps recommending random vectors.
        print("No more recommendations available. Try looking at more humans: ")
        if collect_more_data(5):
            matrix = generate_recs(list_of_humans)
            if not matrix:
                print("Sorry. There are still no recommendations found. How picky are you?")
                print("Here is your 'idea' candidate: ", mean_vector_like)
                return
        else:
            print("Sorry. There's no more data to collect.")
            print("Here is your 'idea' candidate: ", mean_vector_like)
            return
    for human in matrix:
        print(human)
        user_input = input("Do you like them? Yes or No: ")
        if user_input in yes_answers:
            M_like.append(human)
            mean_vector_like = np.array(mean_vector(M_like)) ## Update the mean vector based on the new vector added.
        elif user_input in no_answers:
            M_dislike.append(human)
            mean_vector_dislike = np.array(mean_vector(M_dislike))
        else:
            user_input = input("Respond with Yes or No: ")
    user_input = input("Do you want to see more humans? Yes or No: ")
    while user_input in yes_answers: ## This loop allows the user to keep looking at recommendations.
        new_recs_matrix = generate_recs(list_of_humans)
        if not new_recs_matrix: ## This is in the case that we don't get any recommendations. The program keeps recommending random vectors.
            print("No more suitable recommendations available. Try looking at more humans: ")
            if collect_more_data(5):
                new_recs_matrix = generate_recs(list_of_humans)
                if not new_recs_matrix:
                    print("Sorry. There are still no recommendations found. How picky are you?")
                    print("Here is your 'idea' candidate: ", mean_vector_like)
                    return
            else:
                print("Sorry. There's no more data to collect.")
                print("Here is your 'idea' candidate: ", mean_vector_like)
                return
        give_recs(new_recs_matrix)
        return
    if user_input in no_answers:
        print("Here is your 'ideal' candidate: ", mean_vector_like)
    else:
        user_input = input("Respond with Yes or No: ")

## If there are recommendations to give, call the functions. This block of code also
## gives random vectors in the cases that there are no more recommendations.
if recommendations:
    give_recs(recommendations)
else:
    print("No recommendations found. Try rating more candidates.")
    if collect_more_data(10):
        recommendations = generate_recs(list_of_humans)
        if recommendations:
            give_recs(recommendations)
        else:
            print("Sorry. There are still no recommendations found. How picky are you?")
            print("Here is your 'idea' candidate: ", mean_vector_like)
    else:
        print("Sorry. There's no more data to collect.")
        print("Here is your 'ideal' candidate: ", mean_vector_like)