# Llama2 chatbot

This README will guide you through the setup and usage of the Llama2 chatbot.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/GabrielaMichelon/mychatbot.git
    ```

2. Create a Python virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install Python 3.6 or higher and the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required language models and data. For example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin 

Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.


5. Set up the necessary paths and configurations in your project, including the `DATA_PATH` and `DB_FAISS_PATH` variable and other configurations as per your needs.

6. Run ingest.py file to train the model on your data and store the chunks of text:

    ```bash
    python run ingest.py
    ```

7. Run the model.py file to open the chainlit GUI where you can send your queries:

    ```bash
    chainlit run model.py -w
    ```

## License

This project is based on another [repository](https://github.com/AIAnytime/Llama2-Medical-Chatbot?tab=readme-ov-file#prerequisites).

---

Happy coding with Llama2 chatbot! 🚀
