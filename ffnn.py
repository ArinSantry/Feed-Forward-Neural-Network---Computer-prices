# feed-forward neural network
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv("computer_data.csv")
processXdata = data.iloc[:,:-1]
X = processXdata.apply(LabelEncoder().fit_transform) # make strings into floats so regression works

# data from files
Y = data.iloc[:,-1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.fit_transform(X_test)

reg = MLPRegressor(hidden_layer_sizes=(32,32,32,32,32), activation="relu", random_state=1, max_iter=10000).fit(X_trainscaled, Y_train)

prediction = reg.predict(X_testscaled)

print("MSE: " + str(mean_squared_error(Y_test, prediction, squared=False)))
print("RMSE: " + str(mean_squared_error(Y_test, prediction, squared=True)))
print("R2: " + str(r2_score(Y_test, prediction)))
print()

print('Example predictions:')
df = pd.DataFrame({'Actual': Y_test, 'Predicted': prediction})
print(df)


