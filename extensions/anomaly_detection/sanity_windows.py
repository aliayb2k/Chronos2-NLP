import numpy as np
from chronos2_nlp.data.datasets import load_domain_series
from chronos2_nlp.data.windowing import WindowConfig, make_windows

def main():
    for domain in ["finance_spy", "energy_solar_1D"]:
        sd = load_domain_series(domain)
        v = sd.df["value"].values.astype(np.float32)
        cfg = WindowConfig(context=64, horizon=16, stride=1)
        X, Y = make_windows(v, cfg)
        print(domain, "T=", len(v), "X=", X.shape, "Y=", Y.shape)

if __name__ == "__main__":
    main()