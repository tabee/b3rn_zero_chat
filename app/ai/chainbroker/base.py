"""ContextualCompressionRetriever to retrieve and compress documents."""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter


def pretty_print_docs(docs):
    """Helper function to print out the documents returned by the retriever."""
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" +
          d.page_content for i, d in enumerate(docs)]))


def contextual_compression_embeddings_filter(
        query, embeddings, retriever, similarity_threshold=0.76):
    """The EmbeddingsFilter provides a cheaper and faster option by 
    embedding the documents and query and only returning those 
    documents which have sufficiently similar embeddings to the query."""
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )
    return compression_retriever.get_relevant_documents(query) # returns a list of Document objects


def contextual_compression_document_transformer(
        query, embeddings, retriever, chunk_size=500, chunk_overlap=0):
    """Stringing compressors and document transformers together. 
    We create a compressor pipeline by first splitting our 
    docs into smaller chunks, then removing redundant documents, 
    and then filtering based on relevance to the query.
    Using the DocumentCompressorPipeline we can also easily combine 
    multiple compressors in sequence. Along with compressors we can 
    add BaseDocumentTransformers to our pipeline, which don't perform any 
    contextual compression but simply perform some transformation on a set
    of documents. For example TextSplitters can be used as document 
    transformers to split documents into smaller pieces, and the 
    EmbeddingsRedundantFilter can be used to filter out redundant 
    documents based on embedding similarity between documents. 
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents(query)
    # pretty_print_docs(compressed_docs)
    return compressed_docs
