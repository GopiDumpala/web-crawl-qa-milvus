# Web Crawler and Qa from vectordatabase(milvus) by using llm

## Overview

This project involves developing a web crawler that scrapes data from the NVIDIA CUDA documentation website, processes the data, and provides a search interface using Streamlit.

## Features

- **Web Crawling**: Scrapes data from the NVIDIA CUDA documentation and its sub-links up to 5 levels deep.
- **Data Processing**: Chunks the scraped data, generates embeddings using a sentence transformer model, and stores them in a Milvus vector database.
- **Search Interface**: Provides a search interface using Streamlit, supporting hybrid retrieval and question answering.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/GopiDumpala/cuda-docs-webcrawler.git
    cd cuda-docs-webcrawler
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Web Crawling**: Run the crawler to scrape data:
    ```sh
    python src/cuda_crawler.py
    ```

2. **Data Processing**: Process the scraped data and create embeddings:
    ```sh
    python src/process_data.py
    ```

3. **Run Streamlit App**: Start the Streamlit app to search the CUDA documentation:
    ```sh
    streamlit run src/app.py
    ```

## License

This project is licensed under the MIT License.
