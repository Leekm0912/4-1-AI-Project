import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import metrics
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


def get_accuracy(real_value, estimated_value):
    error_sum = len(real_value)
    for i in range(len(real_value)):
        error = 1 - (estimated_value[i] / real_value[i])
        error_sum -= abs(error)
    return error_sum / len(real_value)


np.random.seed(0)
tf.random.set_seed(0)

if __name__ == '__main__':
    data = pd.read_csv("insurance.csv")
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
    data.drop("region", axis=1, inplace=True)

    # region -> compound
    data["region"] = data["age"] * data["bmi"]
    data["region"] = (data["region"] - data["region"].mean()) / data["region"].std()
    data["bmi"] = (data["bmi"] - data["bmi"].mean()) / data["bmi"].std()
    data["age"] = (data["age"] - data["age"].mean()) / data["age"].std()
    data["sex"] = np_utils.to_categorical(data["sex"], 2)
    data["smoker"] = np_utils.to_categorical(data["smoker"], 2)

    #ax = plt.subplots()
    #ax = sns.violinplot(x="sex", y="charges", hue="smoker", data=data, split=True)
    #ax.set_title("Insurance charges by sex and smoker")
    #plt.show()

    #print(data.groupby("sex")["charges"].mean())

    #scatter = sns.lmplot(x="region", y="charges", data=data, fit_reg=False)
    #cat = sns.lmplot(x="bmi", y="charges", data=data, hue="smoker")
    #plt.show()
    print(data.head())

    #data = data.to_numpy()
    input_c = 5
    x = data.iloc[:, 0:input_c]
    y = data.iloc[:, input_c]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=100)
    print(data.head())
    print(x_test.shape)
    print(data.isnull().sum())
    """
    model = Sequential()
    model.add(Dense(512, input_dim=input_c, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    model_history = model.fit(x_train,
                              y_train,
                              #validation_data=(x_test, y_test),
                              validation_split=0.3,
                              epochs=200,
                              batch_size=100,
                              callbacks=[early_stopping_callback])

    y_prediction = model.predict(x_test).flatten()
    real_charge = y_test.to_numpy()
    for i in range(10):
        label = real_charge[i]
        prediction = y_prediction[i]
        print(f"real charges: {label}, estimated charges: {prediction}")

    #x_len = np.arange(len(y_prediction))
    #plt.plot(x_len, y_prediction, label="estimated", alpha=0.3)
    #plt.plot(x_len, y_test, label="real", alpha=0.3)
    #plt.legend()
    #plt.show()
    print(f"Accuracy: {get_accuracy(real_charge, y_prediction)}")
    print(f"Accuracy: {model.evaluate(x_test, y_test)}")
    """