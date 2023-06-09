"""some base module for webextractor."""
import os
import sys
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
PATH_TO_WORKSPACE = os.getenv(load_dotenv() and "PATH_TO_WORKSPACE")


def _extract_text_from(url, tag):
    """Extract text from a web page."""
    #response = requests.get(url, timeout=1)
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        html = response.text
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["header", "footer", "nav"]):
            script.decompose()
        body = soup.find(tag)
        if body is None:
            return ''
        text = body.get_text()
        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    else:
        return Exception(f"Error {response.status_code} while fetching {url}")


def get_documents_from_sitemap(sitemap_path, sitemap_url_filter):
    """Get documents from a sitemap."""
    sitemap_path = PATH_TO_WORKSPACE + sitemap_path
    print(f"Reading sitemap from {sitemap_path}")
    print(f"Filtering for urls containing {sitemap_url_filter}")
    with open(sitemap_path, 'r', encoding='utf-8') as file:
        xml = file.read()
    raw = xmltodict.parse(xml)
    print(f"Found {len(raw['urlset']['url'])} urls in {sitemap_path}")
    pages = []
    for info in raw['urlset']['url']:
        url = info['loc']
        for wanted_tag_container in ['aside', 'main']:
            if sitemap_url_filter in url:
                pages.append({'text': _extract_text_from(
                    url, wanted_tag_container), 'source': url})
                #print(
                #    f"add {url} :\n {_extract_text_from(url, wanted_tag_container)}\n\n")

    text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")
    return text_splitter.create_documents(docs, metadatas)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print("Usage: python xml-generator.py <sitemap_path> <sitemap_url_filter>")
            sys.exit(1)

        documents = get_documents_from_sitemap(
            sitemap_path=sys.argv[1], sitemap_url_filter=sys.argv[2])
        print(f"Generated {len(documents)} documents")
        print(documents)
    except Exception as e:
        print(f"An error occurred: {e}")
