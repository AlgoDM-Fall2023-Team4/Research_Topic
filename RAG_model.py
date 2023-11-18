#
# Initialize LLM provider
# (google-cloud-aiplatform must be installed)
#

from google.cloud import aiplatform
aiplatform.init(
    project= project_id,
    location='us-east1'

)
#
# Imports
#
from langchain.llms import VertexAI

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import json



##Loading Data

# Read CSV file into a DataFrame
csv_file_path = "/content/drive/MyDrive/output/Smart_Watch_Review.csv"  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

try:
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
 

    # Convert DataFrame to a list of dictionaries, with 'Review' and 'Rating' as variables
    data_list = df.to_dict(orient='records')

    # Specify the output JSON file path
    output_json_file_path = "output.json"

    # Write the list of dictionaries to the output JSON file
    with open(output_json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=2)

    print(f"JSON data has been written to: {output_json_file_path}")

except Exception as e:
    print(f"Error: {e}")
