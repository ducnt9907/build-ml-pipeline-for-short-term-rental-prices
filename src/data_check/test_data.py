import argparse
import logging
import wandb
import pandas as pd
import numpy as np
import scipy.stats


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data, min_price, max_price):
    assert data['price'].between(min_price, max_price).all()


def go(args):

    run = wandb.init(job_type="data_check")
    run.config.update(args)

    csv_path = run.use_artifact(args.csv).file()
    data = pd.read_csv(csv_path)

    ref_path = run.use_artifact(args.ref).file()
    ref_data = pd.read_csv(ref_path)

    test_column_names(data)
    test_neighborhood_names(data)
    test_proper_boundaries(data)
    test_similar_neigh_distrib(data, ref_data, args.kl_threshold)
    test_row_count(data)
    test_price_range(data, args.min_price, args.max_price)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check dataset")


    parser.add_argument(
        "--csv", 
        type=str,
        help="dataset",
        required=True
    )

    parser.add_argument(
        "--ref", 
        type=str,
        help="reference dataset",
        required=True
    )

    parser.add_argument(
        "--kl_threshold", 
        type=float,
        help="Threshold for the KL divergence test on the neighborhood group column",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum accepted price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum accepted price",
        required=True
    )


    args = parser.parse_args()

    go(args)
