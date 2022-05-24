import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

root_path = "data"
book_fol = f"{root_path}/book_train.parquet/"
book_files = book_fol + "*"
book_files = glob.glob(book_files)

trade_fol = f"{root_path}/trade_train.parquet/"
trade_files = trade_fol + "*"
trade_files = glob.glob(trade_files)


def realized_volatility(returns):
    return np.sqrt(np.sum(returns ** 2))


# def remove_outliers(series):
#     cu = 6
#     ser_mean, ser_std = np.mean(series), np.std(series)
#     series = series.where(series <= (ser_mean + cu * ser_std), ser_mean)
#     series = series.where(series >= (ser_mean - cu * ser_std), ser_mean)
#     return series


def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ["time_id", "stock_id"]:
            ret.append(c[0])
        else:
            ret.append(".".join([prefix] + list(c)))
    return ret


book_features = {
    "seconds_in_bucket": ["count"],
    "wap1": [np.mean, np.std, np.sum, np.min, np.max],
    "wap2": [np.mean, np.std, np.sum, np.min, np.max],
    "log_return1": [realized_volatility, np.mean, np.std, np.sum, np.min, np.max],
    "log_return2": [realized_volatility, np.mean, np.std, np.sum, np.min, np.max],
    "bid_ask_spread_p1": [np.mean, np.std, np.sum, np.min, np.max],  # liquidity
    "price_spread": [np.mean, np.std, np.sum, np.min, np.max],
    "bid_ask_spread_v1": [np.mean, np.std, np.sum, np.min, np.max],  # liquidity
    "bid_ask_spread_p2": [np.mean, np.std, np.sum, np.min, np.max],
    "bid_ask_pread_v2": [np.mean, np.std, np.sum, np.min, np.max],
    "bid_spread": [np.mean, np.std, np.sum, np.min, np.max],
    "ask_spread": [np.mean, np.std, np.sum, np.min, np.max],
    "wap_spread": [np.mean, np.std, np.sum, np.min, np.max],
    "bid_size": [np.sum, np.mean, np.std, np.min, np.max],  # liquidity
    "ask_size": [np.sum, np.mean, np.std, np.min, np.max],  # liquidity
    "bid_ask_depth_ratio": [np.sum, np.mean, np.std, np.min, np.max],  # liquidity
    "bid_price1": [np.sum, np.min, np.max],
    "ask_price1": [np.sum, np.min, np.max],
}


trade_features = {
    "seconds_in_bucket": ["count"],
    "price": [realized_volatility, np.mean, np.std, np.sum, np.min, np.max],
    "size": [np.mean, np.std, np.sum, np.min, np.max],
    "order_count": [np.mean, np.std, np.sum, np.min, np.max],
    "trade_volumn": [np.mean, np.std, np.sum, np.min, np.max],
    "weighted_price": [realized_volatility, np.mean, np.std, np.sum, np.min, np.max],
}


def create_book_train_df(path_list):
    list_of_df = []
    for i in tqdm(path_list):
        df = pd.read_parquet(i)
        stock_id = i.split("/")[-1].split("=")[-1]

        # basic features
        ## average price = wap
        df["wap1"] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (
            df.ask_size1 + df.bid_size1
        )
        df["wap2"] = (df.bid_price2 * df.ask_size2 + df.ask_price1 * df.bid_size1) / (
            df.ask_size1 + df.bid_size1
        )

        ## return
        df = df.sort_values(by=["time_id", "seconds_in_bucket"], ascending=[True, True])
        df["previous_wap1"] = df.groupby(["time_id"])["wap1"].shift(1)
        df["previous_wap2"] = df.groupby(["time_id"])["wap2"].shift(1)
        df["log_return1"] = (df["wap1"] / df["previous_wap1"]).apply(
            lambda x: math.log(x)
        )
        df["log_return2"] = (df["wap2"] / df["previous_wap2"]).apply(
            lambda x: math.log(x)
        )

        ## liquidity

        df["bid_ask_spread_p1"] = (
            df["bid_price1"] / df["ask_price1"] - 1
        )  # liquidity (1)
        df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
            (df["ask_price1"] + df["bid_price1"]) / 2
        )  # liquidity (1)
        df["bid_ask_spread_v1"] = abs(
            df["bid_size1"] / df["ask_size1"] - 1
        )  # liquidity (1)

        df["bid_ask_spread_p2"] = (
            df["bid_price2"] / df["ask_price2"] - 1
        )  # liquidity (1)
        df["bid_ask_pread_v2"] = abs(
            (df["bid_size1"] + df["ask_size1"]) / (df["bid_size2"] + df["ask_size2"])
            - 1
        )  # liquidity (1)

        df["bid_ask_depth_ratio"] = (df["bid_size1"] + df["bid_size2"]) / (
            df["ask_size1"] + df["ask_size2"]
        )  # liquidity (1)

        df["bid_size"] = df["bid_size1"] + df["bid_size2"]  # liquidity (2)
        df["ask_size"] = df["ask_size1"] + df["ask_size2"]  # liquidity (2)

        df["bid_spread"] = df["bid_price1"] / df["bid_price2"] - 1  # liquidity (3)
        df["ask_spread"] = df["ask_price1"] / df["ask_price2"] - 1  # liquidity (3)
        df["wap_spread"] = df["wap2"] / df["wap1"] - 1  # liquidity (3)

        # aggregate df
        df_agg = df.groupby("time_id").agg(book_features).reset_index(drop=False)
        df_agg.columns = flatten_name("book", df_agg.columns)
        df_agg["book.bid_ask_spread.overall"] = (
            df_agg["book.bid_price1.amax"] / df_agg["book.ask_price1.amin"] - 1
        )

        #
        df_agg["stock_id"] = stock_id
        df_agg["row_id"] = df_agg["time_id"].apply(
            lambda x: f"{str(stock_id)}-{str(x)}"
        )

        list_of_df = list_of_df + [df_agg]
    return pd.concat(list_of_df)


book_df = create_book_train_df(book_files)
# print(book_df.dtypes)
# book_df.head(10)


def create_trade_train_df(path_list):
    list_of_df = []
    for i in tqdm(path_list):
        df = pd.read_parquet(i)
        stock_id = i.split("/")[-1].split("=")[-1]
        df["stock_id"] = stock_id
        df["trade_volumn"] = df["price"] * df["size"]
        df["weighted_price"] = df["trade_volumn"] / df["size"]
        df["row_id"] = df["time_id"].apply(lambda x: f"{str(stock_id)}-{str(x)}")
        df_agg = df.groupby("time_id").agg(trade_features).reset_index(drop=False)
        df_agg.columns = flatten_name("trade", df_agg.columns)
        df_agg["stock_id"] = stock_id
        df_agg["row_id"] = df_agg["time_id"].apply(
            lambda x: f"{str(stock_id)}-{str(x)}"
        )
        list_of_df = list_of_df + [df_agg]

    return pd.concat(list_of_df)


trade_df = create_trade_train_df(trade_files)
# print(trade_df.dtypes)
# trade_df.head(10)

feature_data = book_df.merge(trade_df, on=["row_id", "time_id", "stock_id"], how="left")

feature_data = feature_data.drop(
    columns=["time_id", "book.seconds_in_bucket.count", "trade.seconds_in_bucket.count"]
)
first_column = feature_data.pop("row_id")
feature_data.insert(0, "row_id", first_column)

target_data = pd.read_csv(f"{root_path}/train.csv")
target_data["row_id"] = (
    target_data["stock_id"].astype(str) + "-" + target_data["time_id"].astype(str)
)

target_data.drop(columns=["stock_id"], inplace=True)
dataset = feature_data.merge(target_data, how="left", on="row_id")
dataset.drop(columns=["time_id", "stock_id"], inplace=True)

# Feature engineering
# replace missing data
means = dataset._get_numeric_data().mean()
dataset = dataset.fillna(means)

# scalling data
name = []
max_value = []
min_value = []

for fe in dataset.columns:
    name.append(fe)
    max_value.append(dataset[fe].max())
    min_value.append(dataset[fe].min())

table = pd.DataFrame(
    list(zip(name, max_value, min_value)), columns=["name", "max", "min"]
)

table.drop(table.tail(1).index, inplace=True)
table.drop(table.head(1).index, inplace=True)
name_over_one = table[(table["max"] > 1) | (table["min"] < 0)]

col_scale = name_over_one["name"].reset_index(drop=True)

### scaling data
scaler = MinMaxScaler()
# scaler = StandardScaler()

# fit the scaler to our data
scaled_df = scaler.fit_transform(dataset[col_scale])

# scale our data
dataset[col_scale] = scaled_df
dataset["stock_id"] = dataset["row_id"].str.split("-", expand=True)[0]

# dataset = pd.read_csv(f"{root_path}/scaled_data.csv")
print(dataset.head())
dataset.to_csv(f"{root_path}/scaled_data.csv", index=False)

