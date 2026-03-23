"""Jane Street Battle Royale - with Real Data Support"""
import polars as pl, numpy as np, os, warnings
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "jane_street_data")

def get_data():
    p = os.path.join(DATA_DIR, "train.parquet")
    if os.path.exists(p):
        print(f"Loading: {p}")
        # Handle partitioned dataset
        if os.path.isdir(p):
            print("  (partitioned dataset)")
            df = pl.scan_parquet(os.path.join(p, "**", "*.parquet")).collect()
        else:
            df = pl.read_parquet(p)
        print(f"  {len(df):,} rows")
        return df, "REAL"
    print("Synthetic data...")
    np.random.seed(42)
    n = 50000
    d = np.random.randint(0, 200, n)
    f = {f"feature_{i:02d}": np.random.randn(n)*(0.5 if i<10 else 1) for i in range(79)}
    t = 0.3*f["feature_00"]+0.2*f["feature_01"]+0.15*f["feature_02"]+0.1*f["feature_03"]+np.random.randn(n)*0.5
    return pl.DataFrame({"date_id":d,"weight":np.random.uniform(0.5,2,n),"responder_6":t,**f}),"SYNTHETIC"

def test_mistral(df):
    r = {"model":"Mistral-Large-675B","status":"FAIL","mse":None}
    try:
        d = df.clone()
        tq = d.group_by("date_id").agg(pl.col("feature_00").quantile(0.85).alias("fq"))
        d = d.join(tq,on="date_id").with_columns(pl.when(pl.col("feature_00")>=pl.col("fq")).then(1).otherwise(0).alias("fq_flag"))
        c = [x for x in d.columns if x.startswith("feature") or x.endswith("flag")]
        X,y = d.select(c).to_numpy(), d["responder_6"].to_numpy()
        Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2,random_state=42)
        m = XGBRegressor(n_estimators=50,max_depth=6,tree_method="hist",n_jobs=-1,random_state=42)
        m.fit(Xt,yt)
        r["mse"],r["status"] = mean_squared_error(yv,m.predict(Xv)),"OK"
    except Exception as e: r["error"] = str(e)
    return r

def test_prod(df):
    r = {"model":"Production-Ready","status":"FAIL","mse":None}
    try:
        d = df.clone()
        tq = d.group_by("date_id").agg(pl.col("feature_00").quantile(0.85).alias("fq"))
        d = d.join(tq,on="date_id").with_columns(pl.when(pl.col("feature_00")>=pl.col("fq")).then(1).otherwise(0).alias("fq_flag"))
        c = [f"feature_{i:02d}" for i in range(79)]+["fq_flag"]
        X,y = d.select(c).to_numpy(), d["responder_6"].to_numpy()
        Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2,random_state=42)
        m = XGBRegressor(n_estimators=100,max_depth=8,learning_rate=0.05,tree_method="hist",n_jobs=-1,random_state=42)
        m.fit(Xt,yt)
        r["mse"],r["status"] = mean_squared_error(yv,m.predict(Xv)),"OK"
    except Exception as e: r["error"] = str(e)
    return r

if __name__ == "__main__":
    print("="*60)
    print("JANE STREET BATTLE ROYALE")
    print("="*60)
    df,dtype = get_data()
    print(f"Data: {dtype}\n")
    results = [test_mistral(df), test_prod(df)]
    for r in results:
        print(f"{r['model']}: {r['status']}" + (f" MSE:{r['mse']:.6f}" if r["mse"] else f" {r.get('error','')[:50]}"))
    valid = [r for r in results if r["mse"]]
    if valid:
        w = min(valid,key=lambda x:x["mse"])
        print(f"\nWinner: {w['model']} (MSE:{w['mse']:.6f})")
