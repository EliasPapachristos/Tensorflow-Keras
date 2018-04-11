#   moon.py
#   Defines a network that can find separate data of moon shapes

#   Imports
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb                 # makes the plot look pretty


# Helper functions
#   plot the moons only on a figure
def plot_moons(pl, X, y):
    # plot half moon for class where y==0
    pl.plot(X[y==0, 0], X[y==0,1], 'ob', alpha=0.5)
    # plot half moon for class where y==1
    pl.plot(X[y==1, 0], X[y==1,1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl
    

def plot_decision_boundary(model, X, y):
    
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    # print("amin:", amin)
    # print("amax:", amax)
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    # make prediction with the model and reshape the output so contourf can plot it
    # print("ab:", ab)
    c = model.predict(ab)
    Z = c.reshape(aa.shape)
    # print("Z shape:", Z.shape)
    # print("Z[0,]:")
    # print(Z[0,])

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_moons(plt, X, y)

    return plt


# Generate some data moons.  Data will be either 0 or 1 and in two "cresent moon" shapes.
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
#pl = plot_moons(plt, X, y)
#pl.show()

# Split the data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the keras model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 1 Hidden Layer Model
#   Simple Sequential model
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="tanh"))
model.add(Dense(2, activation="tanh"))
#   Add a Dense Fully Connected Layer with 1 neuron and the sigmoid activation function
#   return 0 or 1 signifying which moon the predicted value belongs to
model.add(Dense(1, activation="sigmoid"))
#   compile the model.  Minimize crossentopy for a binary.  Maximize accuracy
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
#   fit the model with the data from make_moons
model.fit(X_train, y_train, epochs=100, verbose=0)

#   Get loss and accuracy on test data
eval_result = model.evaluate(X_test, y_test)
#   Print test accuracy
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])

plot_decision_boundary(model, X, y).show()