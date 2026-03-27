from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
  model="text-embedding-3-small",
)

vector = embeddings.embed_query("Hello, world!")

print(vector)
print(len(vector))