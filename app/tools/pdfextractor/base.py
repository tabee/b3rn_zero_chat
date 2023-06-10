"""Module providing functions for extracting documents from pdf files."""
from langchain.document_loaders import PDFPlumberLoader

def _union(lst1, lst2):
    """Function for merging two lists."""
    merged_lst = lst1 + lst2
    return merged_lst

def pdf_extractor_from_single_file(string):
    """Function for creating documents from a single pdf file. 
    Accepts a string to a local file or a url."""
    loader = PDFPlumberLoader(string)
    documents = loader.load()
    return documents

def pdf_extractor_from_files(list_of_urls):
    """Function for creating documents from a list of pdf files. 
    Accepts a list of strings to local files or urls."""
    documents = []
    for url in list_of_urls:
        more_documents = pdf_extractor_from_single_file(url)
        for docs in more_documents:
            docs.metadata['source'] = url
        documents = _union(documents, more_documents)
    return documents
