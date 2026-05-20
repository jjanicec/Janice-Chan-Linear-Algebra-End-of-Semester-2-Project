import numpy as np

## Generate a large set of real n vectors, and each 
## vector represents a human. Although arbitrary for 
## the actual calculations, for the sake of interpretation, 
## the human qualities we are storing are idk
## Each quality is measuerd on a Likert scale from 1 to 5,
## with 5 indicating a lot of that quality and 1 indicating
## very little of that quality.

number_qualities = 5

list_of_humans = []
for i in range 300:
    human = []
    for j in range number_qualities:
        human.append(randint(1,5))
        ## This adds a random score for each quality for each human
    list_of_humans.append(human)

print(list_of_humans)
