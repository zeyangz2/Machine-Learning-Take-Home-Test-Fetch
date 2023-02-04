import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
%matplotlib inline

path = 'data_daily.csv'
df = pd.read_csv(path, header=None, names=['date', 'value'], skiprows=1)

print (df.head())
#pirnt out graph
#x-coordinate is the date, y-coordinate is the count
df['value'].dropna().plot(marker='o', ls='')

#After I printed out the graph for every day's count, I find that the count are increasing linearly.
#Thus, I think that a simple linear regression model is enough for creating a model for this data
#Because the data are perfectly increasing linearly

y=np.array(df['value'].dropna().values, dtype=float)
x=np.array(pd.to_datetime(df['value'].dropna()).index.values, dtype=float)
slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
xf1 = pd.to_datetime(xf1)
yf = (slope*xf)+intercept
print('r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)

f, ax = plt.subplots(1, 1)
ax.plot(xf, yf,label='Linear fit', lw=3)
df['value'].dropna().plot(ax=ax,marker='o', ls='')
plt.ylabel('count prediction')
ax.legend();
#this graph shows the linear model I fit for the data

#I also tried to not use the closed form of linear regression, but gradient descent to show my knowledge of ML.
#However, I think this is not working well since changing a timestamp to string or float causes a lot issues, such as very large or small number
#If I have more time on this, I would transform all dates to some reasonable float number and the issue can be fixed

def costFunction(xVector, yVector, theta):
    inner = np.power(((xVector * theta.T) - yVector), 2)
    return np.sum(inner) / 2
arrayOfOnes = np.ones((len(df['date']),1))[:,0]
## the input vector
xVector = np.column_stack((df['value'].astype(float), arrayOfOnes))

## the vector comprising the target value
yVector = np.matrix(df['value']).T

## initialise theta with some values
theta = np.matrix(np.array([0.00, 0.00]))
costFunction(xVector, yVector, theta)

learningRate = 0.001
iterations = 1000
# stochastic gradient descent
# def gradientDescent(xVector, yVector, theta, learningRate, iterations):
    
    
#     costs = np.zeros(iterations)
    
    
#     m = np.size(theta,1)
#     newTheta = np.matrix(theta).T
    
#     for ite in range(iterations):
       
#         costs[ite] = costFunction(xVector, yVector, newTheta.T)
        
#         for i in range(len(xVector)):
#             currentError = yVector[i,0] - (xVector[i,:] * newTheta)
#             for j in range(m):
#                 term = np.multiply(np.multiply(currentError,xVector[i,j]),learningRate) 
                
#                 newTheta[j,0] = newTheta[j,0] + term
#                 print(newTheta[j,0])
                
#     return newTheta, costs

# newTheta, costs = gradientDescent(xVector, yVector, theta, learningRate, iterations)
# print(newTheta)
