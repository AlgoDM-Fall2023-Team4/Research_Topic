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



#Loading Data

# Read the CSV file into a DataFrame
csv_file_path = '/content/drive/MyDrive/output/Smart_Watch_Review.csv'
df = pd.read_csv(csv_file_path)

# Create a new text file and write the content to it
txt_file_path = '/content/drive/MyDrive/output/output_file.txt'
df.to_csv(txt_file_path, sep='\t', index=False)

print(f'Text file created at: {txt_file_path}')
