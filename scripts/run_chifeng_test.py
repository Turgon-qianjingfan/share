# Reuses the standard Backtester; execute with data/demo/600988_SH_1000d.csv.
from share.data import load_csv
from share.backtest import Backtester
from share.metrics import summarize

def main():
    df=load_csv("data/demo/600988_SH_1000d.csv")
    equity,trades=Backtester(initial_cash=200_000).run(df)
    print(summarize(equity,200_000))
    print("trades=",len(trades))

if __name__=="__main__": main()
