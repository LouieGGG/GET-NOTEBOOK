import pandas as pd
import seaborn as sns

import ml

def main():
    df = sns.load.dataset('iris')
    print(df.head(3))
    print(df.tail(3))
    print()
    print(ml.predict(15))

if __name__ == "__main__":
    main()