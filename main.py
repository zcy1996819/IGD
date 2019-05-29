import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


###########################################TASK1###########################################

def generate_random_numbers(n, mu, sigma, dist="normal"):
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
    elif dist == "uniform":
        return np.random.uniform(mu - sigma/np.sqrt(3) , mu + sigma/np.sqrt(3) , n)
    else:
        raise Exception("The distribution {unknown_dist} is not implemented".format(unknown_dist=dist))

def xIGDTask1(y,n,ordering,type):
    step = 5
    x = np.zeros(n)
    for i in range(0, n):
        x[i] = i + 1
    x = x/n
    iteration = 0
    a,b = 0,0
    plt.subplot(211)
    while iteration < step:
        for i in range(0, n):
            cur = a * x[ordering[i]] + b
            rate = 1/(i+1)
            a = a - rate * (cur - y[ordering[i]]) * x[ordering[i]]
            b = b - rate * (cur - y[ordering[i]])
            plt.plot(a * x + b)
        iteration = iteration + 1

    plt.title("Task1 " + type + " IGD Result History")
    plt.subplot(212)
    plt.title("Task1 " + type + " IGD Result")
    plt.plot(y)
    plt.plot(a * x + b)
    plt.show()
    print("Task1 " + type + " (Mean Error , Sum Error) :  (",
          np.mean(y) - np.mean(a*x+b), np.sum((a*x+b - y)**2)/2, ")")
    return a,b


def IGD_wr_task1(y,type):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    return xIGDTask1(y, n, ordering, "With Replacement " + type)
def IGD_wo_task1(y,type):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    return xIGDTask1(y,n,ordering,"Without Replacement " +type)


n,mu,sigma = 105,0.5,1.0

y1 = generate_random_numbers(n, mu, sigma, "normal")
y1 = np.sort(y1)
y2 = generate_random_numbers(n, mu, sigma, "uniform")
y2 = np.sort(y2)


plt.subplot(211)
plt.title("Normal Distribution")
plt.plot(y1, stats.norm.pdf(y1, mu, sigma))
plt.subplot(212)
plt.title("Uniform Distribution")
plt.plot(y2, stats.uniform.pdf(y2))
plt.show()

print("Task1 : ")

WoTask1Normal_A,WoTask1Normal_B = IGD_wo_task1(y1,"Normal Distribution")
WoTask1Uniform_A,WoTask1Uniform_B = IGD_wo_task1(y2,"Uniform Distribution")

WrTask1Normal_A,WoTask1Normal_B = IGD_wr_task1(y1,"Normal Distribution")
WrTask1Uniform_A,WoTask1Uniform_B = IGD_wr_task1(y2,"Uniform Distribution")

###########################################TASK2###########################################


def xIGDTask2(y,n,ordering,type):
    step = 5
    x = np.zeros(n)
    for i in range(0, n):
        x[i] = i + 1
    x = x/n
    Beta = np.random.uniform(1,2,n)
    mBeta = np.min(1/Beta)
    iteration = 0
    a,b = 0,0
    plt.subplot(211)
    while iteration < step:
        for i in range(0, n):
            cur = a * x[ordering[i]] + b
            rate = 0.95*mBeta*Beta[i]
            a = a - rate * (cur - y[ordering[i]]) * x[ordering[i]]
            b = b - rate * (cur - y[ordering[i]])
            plt.plot(a * x + b)
        iteration = iteration + 1

    plt.title("Task2 "+ type + " IGD Result History")
    plt.subplot(212)
    plt.title("Task2 " + type + " IGD Result")
    plt.plot(y)
    plt.plot(a * x + b)
    plt.show()
    print("Task2 " + type + " (Mean Error , Sum Error) :  (",
          np.mean(y) - np.mean(a * x + b), np.sum(((a * x + b - y) ** 2)*Beta) / 2, ")")
    return a,b




def IGD_wr_task2(y,type):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    return xIGDTask2(y,n,ordering,"With Replacement " +type)

def IGD_wo_task2(y,type):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    return xIGDTask2(y, n, ordering,"Without Replacement " +type)


print("Task2 :")

WoTask2Normal_A,WoTask2Normal_B = IGD_wo_task2(y1,"Normal Distribution")
WoTask2Uniform_A,WoTask2Uniform_B = IGD_wo_task2(y2,"Uniform Distribution")

WrTask2Normal_A,WoTask2Normal_B = IGD_wr_task2(y1,"Normal Distribution")
WrTask2Uniform_A,WoTask2Uniform_B = IGD_wr_task2(y2,"Uniform Distribution")




###########################################TASK3###########################################

def generate_problem_task3(m, n, rho):
    A = np.random.normal(0., 1.0, (m, n))
    x = np.random.random(n) # uniform in (0,1)
    w = np.random.normal(0., rho, m)
    y = A@x + w
    return A, x, y


def xIGDTask3(y,A,m,n,ordering,type,xstar):
    step = 100
    x,w = np.zeros(n),np.zeros(m)
    rate = 0.001
    iteration = 0
    a,b = 0,np.zeros(n)
    plt411 = plt.subplot(411)
    plt.title("Task3 " + type + "History")
    plt412 = plt.subplot(412)
    plt.title("Task3 " + type + "||x-x*||")

    while iteration < step:
        for i in range(0, m):
            orderIdx = ordering[i]
            cur = np.matmul(A[orderIdx] ,x) + w
            for j in range(0,n):
                x[j] = x[j] - rate * A[orderIdx,j] * (cur[orderIdx] - y[orderIdx])
            w[orderIdx] = w[orderIdx] - rate*(cur[orderIdx] - y[orderIdx])
            if iteration < 10:
                plt411.plot(np.matmul(A,x) + w)
                plt412.plot(x-xstar)
        iteration = iteration + 1
    plt.subplot(413)
    plt.title("Task3 " + type + "Y")
    plt.plot(y)
    plt.subplot(414)
    plt.title("Task3 " + type + "Predicted (A*x + w)")
    plt.plot(np.matmul(A,x) + w)
    plt.show()
    print("Task3 " + type +"Y-(A*x + w) (Mean Error , Sum Error ):"
          ,np.mean(y - np.matmul(A,x) + w),np.sum(y - np.matmul(A,x) + w))
    return x,w





def IGD_wr_task3(y, A , xstar):
    m, n = len(y), len(A.T)
    ordering = np.random.choice(m, m, replace=True)
    return xIGDTask3(y, A, m, n, ordering, "With Replacement ",xstar)

def IGD_wo_task3(y, A, xstar):
    m,n = len(y),len(A.T)
    ordering = np.random.choice(m, m, replace=False)
    return xIGDTask3(y, A, m, n, ordering, "Without Replacement ",xstar)


print("Task3 :")
m,n,rho = 200,100,0.01
A, xstar, y = generate_problem_task3(m, n, rho)
Wo_x, Wo_w = IGD_wo_task3(y, A , xstar)
Wr_x, Wr_w = IGD_wr_task3(y, A , xstar)