"""some base module for sitemap-generator"""
import datetime
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from lxml import etree

PATH_TO_WORKSPACE = os.getenv(load_dotenv() and "PATH_TO_WORKSPACE")


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.3'
}


def crawl(url, visited, base_url):
    """crawl the url and extract the links"""
    # Check if the URL has already been visited
    if url in visited:
        return []

    # Send an HTTP request to the URL
    response = requests.get(
        url, headers=headers, timeout=5
    )  # Zeile 13 aufgeteilt

    # Check if the response is successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the links from the HTML page
        links = []

        # Find all the 'a' tags on the page
        a_tags = soup.find_all('a')

        # Check if any 'a' tags were found
        if a_tags:
            for link in a_tags:
                href = link.get('href')
                if href and href.startswith('/'):
                    # Prefix the link with the base URL
                    href = f"{base_url}{href}"
                if href and href.startswith(base_url):
                    links.append(href)
        return links

    return []  # Konsistente Rückgabewerte gewährleisten


def build_sitemap(url, sitemap, visited, base_url):
    """build sitemap"""
    # Get the path of the URL
    path = url.replace(base_url, '')

    # Check if the path has already been visited
    if path in visited:
        return

    # Print the URL that is being crawled
    print(url)

    # Add the URL to the sitemap and mark the path as visited
    sitemap.append(url)
    visited.add(path)

    # Crawl the URL and extract the links
    links = crawl(url, visited, base_url)

    # Follow the links and add them to the sitemap
    for link in links:
        build_sitemap(link, sitemap, visited, base_url)


def make_sitemap(base_url='https://www.example.ch', filename='example_ch'):
    """make sitemap need the base URL of the website you want to crawl and a name for the file."""

    print(f"make sitemap for {base_url}")

    now = datetime.datetime.now()

    # Format the date and time as a string in the desired format
    lastmod = now.strftime("%Y-%m-%d")

    sitemap = []
    visited = set()

    # Start building the sitemap from the base URL
    build_sitemap(base_url, sitemap, visited, base_url)

    # Create the root element of the sitemap
    root = etree.Element(
        'urlset', xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add the URLs to the sitemap as 'url' elements
    for url in sitemap:
        url_element = etree.SubElement(root, 'url')
        etree.SubElement(url_element, 'loc').text = url
        etree.SubElement(url_element, 'lastmod').text = lastmod

    filepath = f'{PATH_TO_WORKSPACE}/data/sitemaps/sitemap_{filename}.xml'

    # Write the XML declaration to the beginning of the file
    with open(filepath, 'w', encoding="utf8") as file:
        file.write("<?xml version='1.0' encoding='UTF-8'?>\n")

    # Write the sitemap to a file as XML
    with open(filepath, 'ab') as file:
        file.write(etree.tostring(root, pretty_print=True))


if __name__ == "__main__":
    NAME = '_bee_gu_ch'
    URL = 'https://www.bee-gu.ch'
    make_sitemap(URL, NAME)
