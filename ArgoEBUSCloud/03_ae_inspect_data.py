import pandas as pd
import matplotlib.pyplot as plt
import s3fs

def inspect_processed_data():
    # 1. Define the S3 path where your cloud run saved the results
    s3_path = "s3://argo-ebus-project-data-abm/processed_ebus_2015.parquet"
    
    print(f"📡 Downloading processed results from {s3_path}...")
    
    # 2. Read the Parquet folder directly into a Pandas DataFrame
    # s3fs (which we just fixed!) handles the security handshake automatically
    df = pd.read_parquet(s3_path)
    
    # 3. Quick Data Audit
    print("\n📊 Dataset Summary:")
    print(df.info())
    print("\n🔍 First 5 Rows:")
    print(df.head())

    # 4. Simple Visualization: OHC Trends over Time
    # We aggregate all bins to see the average heat content per month in 2015
    monthly_trend = df.groupby('time_bin')['ohc_per_m'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_trend['time_bin'], monthly_trend['ohc_per_m'], marker='o', linestyle='-', color='teal')
    
    plt.title('2015 Ocean Heat Content Trend (EBUS Region)', fontsize=14)
    plt.xlabel('Days since 1999-01-01', fontsize=12)
    plt.ylabel('Average OHC per Meter (J/m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot locally so you can view it
    plt.savefig('../AEResults/aeplots/ebus_2015_trend.png')
    print("\n📈 Plot saved as 'ebus_2015_trend.png'")
    plt.show()

if __name__ == "__main__":
    inspect_processed_data()