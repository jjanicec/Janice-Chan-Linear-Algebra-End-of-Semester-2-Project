import numpy as np
import random

## Generate a large set of real n vectors, and each vector represents a human.
## Although arbitrary for the actual calculations, for the sake of interpretation, 
## the human qualities we are storing will come from a chosen list of human qualities.
## Each quality is measuerd on a Likert scale from 1 to 5, with 5 indicating a lot
## of that quality and 1 indicating very little of that quality.

user_input = input("How many human qualities would you like to look at?")
if not(user_input.isdigit()):
    user_input = input("Please enter a whole number value:")
if int(user_input) > 10:
    user_input = input("Please enter a value less than or equal to 10:")
if int(user_input) <= 0:
    user_input = input("Please enter a value greater than 0:")
number_qualities = int(user_input)

list_of_qualities = ["Humor","Empathy","Intelligence","Extroversion","Athleticism","Honesty","Confidence","Aggressiveness","Selfishness","Wealth"]
text = "The human qualities that will be represented by each vector are: "
for word in list_of_qualities[:(number_qualities-1)]:
    text += word
    text += ", "
print(text)

## Note that this is a list of vectors, not a matrix.
list_of_humans = []
for i in range(150):
    human = []
    for j in range(number_qualities):
        human.append(random.randint(1,5)) ## This adds a random score for each quality for each human.
    list_of_humans.append(human)

## Make matrices to store the human vectors in, based on user input.
M_like = []
M_dislike = []

## These lists allow the user to type "yes" and "no" however they want.
yes_answers = ["yes","YES","Yes","y"]
no_answers = ["no","NO","No","n"]

while len(M_dislike) < 5 or len(M_like) < 5:
    for candidate_trial_0 in range(5):
        random_human = list_of_humans[random.randint(0,(len(list_of_humans)-1))]
        print(random_human)
        user_input = input("Do you like them? Yes or No:")
        if user_input in yes_answers:
            M_like.append(random_human)
            list_of_humans.remove(random_human)
        elif user_input in no_answers:
            M_dislike.append(random_human)
            list_of_humans.remove(random_human)
        else:
            user_input = input("Respond with Yes or No:")

## Find the average vector for both sets. The idea is that for qualities
## the user does not care about, they should be random, and the qualities 
## the user does care about should follow a pattern.

## The technique we will use is finding the principle eigenvector of the
## covariance matrix. We will mean-center the data as well.
def mean_vector(matrix):
    mean_vect = np.mean(matrix, axis = 0)
    return mean_vect

def lin_reg(matrix):
    mean = mean_vector(matrix)
    centered = matrix - mean
    n = len(matrix)
    covariance_matrix = np.matmul(centered.T, centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    max_index = np.argmax(eigenvalues)
    line = eigenvectors[:, max_index]
    line = line / np.linalg.norm(line)
    return line

def orth_distance(vector, mean, line):
    vector = np.array(vector)
    line = np.array(line)
    projection = mean + np.dot(vector - mean, line) * line
    orth = vector - projection
    orth_dist = np.linalg.norm(orth)
    return orth_dist

## Calculate the distance between the vector the program is to recommend and the
## mean like and dislike vectors. Then calculate a "score" based on those distances.
mean_vector_like = mean_vector(M_like)
mean_vector_dislike = mean_vector(M_dislike)

def calculate_score(recommended):
    distance_like = orth_distance(recommended, mean_vector_like, lin_reg(M_like))
    distance_dislike = orth_distance(recommended, mean_vector_dislike, lin_reg(M_dislike))
    score = distance_like / distance_dislike
    return score

## If the human fits a certain criteria, then recommend them to the user.
def generate_recs(matrix):
    recommendations = []
    while len(recommendations) < 10:
        for human in matrix:
            score = calculate_score(human)
            if score <= 0.5:
                recommendations.append(human)
                matrix.remove(human)
    return recommendations

recommendations = generate_recs(list_of_humans)

def give_recs(matrix):
    for human in matrix:
        print(human)
        user_input = input("Do you like them? Yes or No:")
        if user_input in yes_answers:
            M_like.append(human)
        elif user_input in no_answers:
            M_dislike.append(random_human)
        else:
            user_input = input("Respond with Yes or No:")

user_input = input("Do you want to see more humans? Yes or No:")
while user_input in yes_answers:
    give_recs(recommendations)
if user_input in no_answers:
    print("Here is your 'ideal' candidate:", mean_vector(M_like))
else: 
    user_input = input("Respond with Yes or No:")