import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader

import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGIONS"],
)
INDEX_NAME = "gst-demo-index"


def ingest_docs()->None:
    loader = PyPDFDirectoryLoader("./gst2")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")


# loader = BSHTMLLoader("./python.langchain.com/index.html")
  # loader=  DirectoryLoader("./python.langchain.com/", glob="**/*.html")





    #print(len(documents))
   #how many documents were converted to chunks
   #convert URLs to https

        #for doc in documents:
       #new_url = doc.metadata["source"]
       #new_url = new_url.replace("langchain-docs", "https:/")
       #doc.metadata.update({"source": new_url})
       #print(new_url)



if __name__ == "__main__":
    ingest_docs()

