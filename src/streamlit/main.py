import streamlit as st

import matplotlib.pyplot as plt
from google.cloud import aiplatform

from langchain.llms import VertexAI
from google.oauth2 import service_account
from google.cloud import bigquery
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

st.set_option('deprecation.showPyplotGlobalUse', False)

keyfile_path = 'charged-formula-405300-4a513f476167.json'

project_id = 'charged-formula-405300'
dataset_id = 'rag_llm'
table_id = 'attributes'

sqlalchemy_url = f'bigquery://{project_id}/{dataset_id}?credentials_path={keyfile_path}'
aiplatform.init(
    project=f'{project_id}',
    location='us-east1'
)

keyfile_path = 'charged-formula-405300-4a513f476167.json'

credentials = service_account.Credentials.from_service_account_file(
    keyfile_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bq_client = bigquery.Client(credentials= credentials, project=project_id)

db = SQLDatabase.from_uri(sqlalchemy_url)

_DEFAULT_TEMPLATE = """Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

Determine which tables need to be joined before generating the SQL query. Limit the number of rows in the SQL result by {top_k}.

Question: {input}"""

custom_prompt = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "top_k"], template=_DEFAULT_TEMPLATE
)

llm = VertexAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm,
                                     db,
                                     top_k=3,
                                     verbose=True,
                                     prompt=custom_prompt,
                                     return_intermediate_steps=True)
st.title('Relational Querying')
input = st.text_input('Enter your question here!')

if input:
    result = db_chain(input)
    st.write("Answer: " + result['result'])

positive, negative, neutral = (437, 529, 30)

data = {'Positive': positive, 'Neutral': neutral, 'Negative': negative}

st.title('Sentiment Analysis Results')
st.bar_chart(data)




from wordcloud import WordCloud

def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    st.pyplot()

query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"

df = bq_client.query(query).to_dataframe()


st.title("Review Word Cloud Generator")

df['liked'] = df['liked'].astype(str)
df['disliked'] = df['disliked'].astype(str)

liked_text = ' '.join(df['liked'])
st.title(":green[Liked]")
generate_word_cloud(liked_text, 'Liked Word Cloud')

disliked_text = ''.join(df['disliked'])
st.title(":red[Disliked]")
generate_word_cloud(disliked_text, 'Disliked Word Cloud')






