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

##Question Answering Using MapReduce
# Load data from JSON file
json_file_path = "output.json"  # Replace with the path to your JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)
print(df.head())


# Combine reviews and ratings into a single text document
combined_reviews = '\n'.join(f"Rating: {rating}\nReview: {review}" for rating, review in zip(df['Rating'], df['Review']))

# Create a text file with the combined reviews and ratings
combined_reviews_file_path = "combined_reviews_with_ratings.txt"
with open(combined_reviews_file_path, 'w') as file:
    file.write(combined_reviews)


# Initialize the language model
llm = VertexAI(temperature=0.7)  # Adjust temperature as needed

# Load the input document
loader = TextLoader(combined_reviews_file_path)
documents = loader.load()

# # Splitting
# text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# print(f'The input document has been split into {len(texts)} chunks\n')

#
# Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0, separators=["Rating"])
texts = text_splitter.split_documents(documents)
print(f'The input document has been split into {len(texts)} chunks\n')
print(texts[0])


