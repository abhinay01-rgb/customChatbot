#!pip install duckduckgo_search langchain
from duckduckgo_search import DDGS
from langchain.document_loaders import WebBaseLoader
import re
ddgs = DDGS()


x = None  # Initialize x as a global variable

def link(link):
    global x  # Declare x as a global variable
    x = link
    print(x)

#query = "what is data matics?"
def get_search_result(query):
    global x  # Declare x as a global variable
    print(f"x is {x}")

    results = ddgs.text(f"site:{x} {query}")
    #results = ddgs.text(f"site:datamatics.com {query}")
    url = []
    text =""
    count = 0
    for result in results:
        #print(result)
        #print(result['href'])
        link = result['href']
         # Remove links to images and various files
        if (
            link.endswith(".png")
            or link.endswith(".json")
            or link.endswith(".txt")
            or link.endswith(".svg")
            or link.endswith(".ipynb")
            or link.endswith(".jpg")
            or link.endswith(".pdf")
            or link.endswith(".mp4")
            or "mailto" in link
            or len(link) > 300
        ):
            continue
        url.append(result['href'])
        count += 1
        if count == 2:
           break
    #print(url)


    # loader = WebBaseLoader(url)
    # documents = loader.load()
    # print(documents)
    # text = " ".join(map(str, documents))
    # #text = re.sub(r'\n+', ' ', text)
    # #text = re.sub('\\n', ' ', text)
    # print(text)

    loader = WebBaseLoader(url)
    documents = loader.load()
    #print(documents)
    page_content = documents[0].page_content
    text = " ".join(page_content.split())
    page_content2 = documents[0].metadata
    finale_text = text+" "+str(page_content2)
    print(finale_text)
    return finale_text
#print(get_search_result(query))