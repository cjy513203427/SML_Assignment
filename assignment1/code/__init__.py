# -*- encoding: utf-8 -*-
'''
@File    :   rosenbrock_tutorial.py.py    
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/5/17 16:17   Jonas           None
'''
 
import numpy as np
import matplotlib.pyplot as plt

# State space
states = ["Mutated","Unmutated"]

# Possible events
transitionName = [["UM","UU"],["MU","MM"]]

# Probilistic matrix
transitionMatrix = [0.446,0.554]



def activity_forecast(days):
    # Choose initial state
    activityToday = "Unmutated"
    print("Start state: " + activityToday)
    # Initial state list
    activityList = [activityToday]
    prob_list = []
    i = 0
    # Calculate the probability of activityList
    prob = 1
    while i != days:
        if activityToday == "Unmutated":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix)
            if change == "UM":
                prob = prob * 0.446
                activityToday = "Mutated"
                activityList.append("Mutated")
                prob_list.append(prob)
                pass
            elif change == "UU":
                prob = prob * 0.554
                activityList.append("Unmutated")
                prob_list.append(prob)
        elif activityToday == "Mutated":
            change = np.random.choice(transitionName[1], replace=True, p=transitionMatrix)
            if change == "MU":
                prob = prob * 0.446
                activityToday = "Unmutated"
                activityList.append("Unmutated")
                prob_list.append(prob)
                pass
            elif change == "MM":
                prob = prob * 0.554
                activityList.append("Mutated")
                prob_list.append(prob)
        i += 1
    print("Possible states: " + str(activityList))
    print("End state after "+ str(days) + " days: " + activityToday)
    print("Probability of the possible sequence of states: " + str(prob))

    x = np.arange(0, 18, 1)
    prob_list = np.array(prob_list)
    plt.plot(x,prob_list)
    plt.xlabel("days")
    plt.ylabel("probability")
    plt.show()

# predict states after 18 days
activity_forecast(18)
