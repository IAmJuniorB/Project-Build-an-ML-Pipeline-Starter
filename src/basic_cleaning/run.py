#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning", group="cleaning", save_code=True)
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("Cleaning data")
    
    # Ensure price is float
    df['price'] = df['price'].astype(float)
    
    # Log price range before filtering
    logger.info(f"Prices before filtering - Min: {df['price'].min()}, Max: {df['price'].max()}")
    
    # Filter by price
    logger.info(f"Filtering prices between {args.min_price} and {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    # Log price range after filtering
    logger.info(f"Prices after filtering - Min: {df['price'].min()}, Max: {df['price'].max()}")
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Filter by longitude and latitude
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned data")
    output_path = "clean_sample.csv"
    df.to_csv(output_path, index=False)

    logger.info("Logging artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact (e.g., sample.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact (e.g., clean_sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., clean_data)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price threshold for filtering properties",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price threshold for filtering properties",
        required=True
    )

    args = parser.parse_args()

    go(args)