import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('./Q1.5/output/predict/SC01_prediction.csv', header=None)
    df = np.array(df)

    plt.figure()
    x = df[:24, 2]
    y = df[:24, 1]

    plt.title("Prediction")
    plt.xlabel("hour")
    plt.ylabel("Number of Order")

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
