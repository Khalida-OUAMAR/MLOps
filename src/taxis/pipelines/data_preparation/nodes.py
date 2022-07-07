import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data(df: pd.DataFrame) -> None :
    new_df = pd.DataFrame()
    
    print(df.columns)
    
    new_df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    new_df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    new_df["VendorID"] = pd.to_numeric(df["VendorID"])
    new_df["total_amount"] = pd.to_numeric(df["total_amount"])
    new_df["PULocationID"] = pd.to_numeric(df["PULocationID"])
    new_df["DOLocationID"] = pd.to_numeric(df["DOLocationID"])
    new_df["trip_distance"] = pd.to_numeric(df["trip_distance"])
    
    
    
    print(new_df.head(2))
    print(new_df.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(new_df[["tpep_pickup_datetime",
            "tpep_dropoff_datetime", "VendorID", "total_amount", "PULocationID", "DOLocationID", "trip_distance"]],
                                                        new_df["tpep_dropoff_datetime"],test_size=0.2)
    
    cat_attribs = ["tpep_pickup_datetime","tpep_dropoff_datetime", "VendorID", "total_amount", "PULocationID", "DOLocationID", "trip_distance"]
    full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

    encoder = full_pipeline.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)



    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_test,
        test_y=y_test
    )