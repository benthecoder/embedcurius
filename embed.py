import json
import logging

import numpy as np
import pandas as pd
import requests
import tiktoken
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"  # replace with your OpenAI API key
CURIUS_ID = "<YOUR_CURIUS_ID>"  # replace with your Curius user ID

# Maximum number of tokens for OpenAI API embedding model
MAX_TOKENS = 8192


def get_curius_links(curius_id):
    """Get links from the Curius API for a given user ID."""
    url = f"https://curius.app/api/users/{curius_id}/searchLinks"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure HTTP errors raise an exception
        data = response.json()
        return data["links"]
    except requests.RequestException as e:
        logging.error(f"Failed to fetch links: {e}")
        return []


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string using TikToken for estimation."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_text(text, max_tokens=MAX_TOKENS, estimated_avg_token_size=2.5):
    """Truncates text to ensure it fits within a specified token limit."""
    max_chars = int(max_tokens * estimated_avg_token_size)
    if len(text) <= max_chars:
        return text
    else:
        return text[:max_chars]


def sanitize_text(text):
    """Sanitizes text to remove newlines and excess whitespace."""
    return text.replace("\n", " ").replace("\r", " ").strip()


def generate_embeddings_and_metadata(client, links):
    """Generate embeddings and metadata for a list of links."""
    embeddings = []
    metadata = []

    for i in tqdm(range(0, len(links), 500), desc="Chunk Progress"):
        chunk = links[i : i + 500]
        chunk_json = [
            json.dumps(
                {
                    "title": link.get("title", ""),
                    "snippet": truncate_text(link.get("snippet", "")),
                }
            )
            for link in chunk
        ]

        try:
            response = client.embeddings.create(
                input=chunk_json, model="text-embedding-3-small"
            )
            chunk_embeddings = np.array([item.embedding for item in response.data])
            embeddings.extend(chunk_embeddings)

            chunk_metadata = [
                {
                    "title": sanitize_text(link.get("title", "")),
                    "date": link.get("createdDate", ""),
                    "link": link["link"],
                }
                for link in chunk
            ]
            metadata.extend(chunk_metadata)
        except Exception as e:
            logging.error(f"Error occurred while generating embeddings: {e}")
            return None, None

    return embeddings, metadata


def save_to_tsv(
    embeddings,
    metadata,
    vectors_filename="vectors.tsv",
    metadata_filename="metadata.tsv",
):
    """Save embeddings and metadata to TSV files."""
    pd.DataFrame(embeddings).to_csv(
        vectors_filename, sep="\t", index=False, header=False
    )
    pd.DataFrame(metadata).to_csv(metadata_filename, sep="\t", index=False)
    logging.info("Embedding and metadata files saved successfully.")


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    links = get_curius_links(CURIUS_ID)
    if not links:
        logging.error("No links fetched. Exiting.")
        return

    logging.info(f"fetched {len(links)} links in total.")
    embeddings, metadata = generate_embeddings_and_metadata(client, links)

    if not embeddings or not metadata:
        logging.error("No embeddings or metadata generated. Exiting.")
        return

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Number of embeddings does not match number of metadata entries."
        )

    save_to_tsv(embeddings, metadata)


if __name__ == "__main__":
    main()
