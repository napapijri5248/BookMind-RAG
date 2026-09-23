from langchain_community.retrievers import ArxivRetriever

# create retriever component
retriever=ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

# query arxiv
docs=retriever.invoke('large language models')

#print resuts

for i,doc in enumerate(docs):
    print(f"Result {i+1} :")
    print('Title : ',doc.metadata.get('Title'))
    print('Authors : ',doc.metadata.get('Authors'))
    print('Summary : ',doc.page_content[:500]) # print first 500
# --------------------------------------------------------------------------
# import arxiv
# from langchain_core.documents import Document


# client = arxiv.Client()

# search = arxiv.Search(
#     query="large language models",
#     max_results=2,
#     sort_by=arxiv.SortCriterion.Relevance
# )

# docs = []

# for result in client.results(search):
#     doc = Document(
#         page_content=result.summary,
#         metadata={
#             "Title": result.title,
#             "Authors": [author.name for author in result.authors],
#             "Published": result.published.isoformat(),
#             "PDF_URL": result.pdf_url,
#             "Entry_ID": result.entry_id
#         }
#     )

#     docs.append(doc)


# for i, doc in enumerate(docs):
#     print(f"\nResult {i + 1}:")
#     print("Title:", doc.metadata["Title"])
#     print("Authors:", doc.metadata["Authors"])
#     print("Summary:", doc.page_content[:500])
#     print("PDF URL:", doc.metadata["PDF_URL"])
# -------------------------------------------------------------
# import requests
# import xml.etree.ElementTree as ET

# from langchain_core.documents import Document


# # ArXiv API URL (HTTPS)
# url = "https://export.arxiv.org/api/query"

# params = {
#     "search_query": "all:large language models",
#     "start": 0,
#     "max_results": 2,
#     "sortBy": "relevance",
#     "sortOrder": "descending"
# }

# response = requests.get(
#     url,
#     params=params,
#     timeout=30
# )

# response.raise_for_status()

# root = ET.fromstring(response.text)

# namespace = {
#     "atom": "http://www.w3.org/2005/Atom"
# }

# docs = []

# for entry in root.findall("atom:entry", namespace):

#     title = entry.find("atom:title", namespace).text.strip()

#     summary = entry.find("atom:summary", namespace).text.strip()

#     published = entry.find("atom:published", namespace).text

#     entry_id = entry.find("atom:id", namespace).text

#     authors = [
#         author.find("atom:name", namespace).text
#         for author in entry.findall("atom:author", namespace)
#     ]

#     pdf_url = None

#     for link in entry.findall("atom:link", namespace):
#         if link.attrib.get("title") == "pdf":
#             pdf_url = link.attrib.get("href")

#     doc = Document(
#         page_content=summary,
#         metadata={
#             "Title": title,
#             "Authors": authors,
#             "Published": published,
#             "PDF_URL": pdf_url,
#             "Entry_ID": entry_id
#         }
#     )

#     docs.append(doc)


# for i, doc in enumerate(docs):

#     print(f"\nResult {i + 1}:")
#     print("Title:", doc.metadata["Title"])
#     print("Authors:", doc.metadata["Authors"])
#     print("Summary:", doc.page_content[:500])
#     print("PDF URL:", doc.metadata["PDF_URL"])