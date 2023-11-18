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


