from category_encoders import CountEncoder, TargetEncoder

from created_column_processing import process_created
from features_column_pipeline import prepare_features
from location_processing import add_distances_to_landmarks
from street_adress_column_processing import prepare_street_addresses


def columns_processing_pipeline(df_train, df_val, df_test):

    df_train, df_val, df_test = prepare_features(
        df_train,
        df_val,
        df_test,
        min_freq=3,
        corr_threshold=0.9,
        drop_original_cols=[
            "photos",
            "listing_id",
            "interest_level",
            "description",
            "display_address",
        ],
    )

    df_train, df_val, df_test = prepare_street_addresses(df_train, df_val, df_test)

    df_train, df_val, df_test = process_created(
        df_train, df_val, df_test, created_col="created", drop_original=True
    )

    df_train, df_val, df_test = add_distances_to_landmarks(df_train, df_val, df_test)

    target_enc = TargetEncoder(
        cols=["street_clean"],
        handle_unknown="value",
        handle_missing="value",
        smoothing=10.0,
    )
    df_train = target_enc.fit_transform(df_train, df_train["price"])
    df_val = target_enc.transform(df_val)
    df_test = target_enc.transform(df_test)

    for df in (df_train, df_val, df_test):
        df.drop(columns=["street_clean"], inplace=True, errors="ignore")

    count_enc = CountEncoder(
        cols=["manager_id", "building_id"],
        handle_unknown="value",
        handle_missing="value",
        min_group_size=5,
    )
    count_enc.fit(df_train)
    df_train = count_enc.transform(df_train)
    df_val = count_enc.transform(df_val)
    df_test = count_enc.transform(df_test)

    for df in (df_train, df_val, df_test):
        df.drop(columns=["manager_id", "building_id"], inplace=True, errors="ignore")

    for df in (df_train, df_val, df_test):
        feature_cols = [c for c in df.columns if c != "price"]
        df[feature_cols] = df[feature_cols].fillna(0)

    return df_train, df_val, df_test
