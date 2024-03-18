import zipfile

import streamlit as st
from openai import OpenAI

from embed import generate_embeddings_and_metadata, get_curius_links, save_to_tsv

st.title("embedcurius")
st.markdown(
    "Generate embeddings for your [curius](https://curius.app/benedict-neo) links to visualize on [Embedding Projector](https://projector.tensorflow.org/)."
)

# enter openai key
openai_key = st.sidebar.text_input("Enter your OpenAI API key", placeholder="sk-...")

# Input for Curius user ID
user_id = st.text_input(
    "Enter your Curius user ID",
    help="where to find this? see https://github.com/benthecoder/embedcurius",
)

# Initialize session state
if "links_processed" not in st.session_state:
    st.session_state["links_processed"] = False

# Button to generate embeddings
if st.button("Generate Embeddings"):
    try:
        client = OpenAI(api_key=openai_key)
        user_id = int(user_id)
        st.session_state["links_processed"] = True
        links = get_curius_links(user_id)

        st.info(f"You have a total of {len(links)} links.")

        with st.spinner("Generating embeddings..."):
            embeddings, metadata = generate_embeddings_and_metadata(client, links)

        if not embeddings or not metadata:
            st.error(
                "No embeddings or metadata generated. Did you provide an OpenAI API key?"
            )
            st.stop()

        st.success("Embeddings generated successfully.")
        if len(embeddings) != len(metadata):
            st.error("Number of embeddings does not match number of metadata entries.")
            st.stop()

        st.info("saving embeddings and metadata to TSV files...")
        save_to_tsv(embeddings, metadata)
    except ValueError:
        st.error("Please enter a valid user ID.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Provide a link to TensorFlow Projector and download button
if st.session_state["links_processed"]:
    # Compress TSV files into a ZIP file
    with zipfile.ZipFile("embeddings.zip", "w") as zipf:
        zipf.write("vectors.tsv")
        zipf.write("metadata.tsv")

    # Provide download button for ZIP file
    with open("embeddings.zip", "rb") as file:
        st.download_button(
            label="Download Embeddings",
            data=file,
            file_name="embeddings.zip",
            mime="application/zip",
        )
