import unittest
from unittest import mock

import sys
sys.path.insert(0, '')

from web_scraper_rag import (
    build_retriever,
    build_rag_chain,
    run_query,
)

class DummyDoc:
    def __init__(self, text):
        self.page_content = text

class LoaderMock:
    def __init__(self, pages):
        self._pages = pages

    def load(self):
        # Return a list of Document-like objects
        return [mock.Mock(page_content=p) for p in self._pages]

class TestEdgeCases(unittest.TestCase):
    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_multiple_urls(self, mock_embeddings, mock_chroma, mock_loader):
        # Simulate loader loading multiple pages across URLs
        mock_loader.return_value.load.return_value = [mock.Mock(page_content='page1'), mock.Mock(page_content='page2')]
        mock_chroma.from_documents.return_value.as_retriever.return_value = mock.Mock(get_relevant_documents=lambda q: [mock.Mock(page_content='page1 content')])

        retriever = build_retriever(['https://a.com', 'https://b.com'])
        docs = retriever.get_relevant_documents('q')
        self.assertTrue(len(docs) > 0)

    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_large_documents(self, mock_embeddings, mock_chroma, mock_loader):
        # Simulate a very large page_content to test splitter handling
        large_text = 'x' * 20000
        mock_loader.return_value.load.return_value = [mock.Mock(page_content=large_text)]
        # We don't need Chroma to actually index; return a retriever
        mock_chroma.from_documents.return_value.as_retriever.return_value = mock.Mock(get_relevant_documents=lambda q: [mock.Mock(page_content=large_text[:100])])

        retriever = build_retriever(['https://large.example'])
        docs = retriever.get_relevant_documents('q')
        self.assertTrue(len(docs) > 0)

    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_invalid_html(self, mock_embeddings, mock_chroma, mock_loader):
        # Simulate loader returning malformed / invalid HTML content
        mock_loader.return_value.load.return_value = [mock.Mock(page_content='<html><bad>>')]
        mock_chroma.from_documents.return_value.as_retriever.return_value = mock.Mock(get_relevant_documents=lambda q: [mock.Mock(page_content='malformed content')])

        retriever = build_retriever(['https://invalid.html'])
        docs = retriever.get_relevant_documents('q')
        self.assertTrue(len(docs) > 0)

    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_loader_permission_timeout(self, mock_embeddings, mock_chroma, mock_loader):
        # Simulate loader throwing permission or timeout errors
        mock_loader.return_value.load.side_effect = PermissionError('403')
        with self.assertRaises(PermissionError):
            _ = build_retriever(['https://forbidden.example'])

        mock_loader.return_value.load.side_effect = TimeoutError('timed out')
        with self.assertRaises(TimeoutError):
            _ = build_retriever(['https://slow.example'])

if __name__ == '__main__':
    unittest.main()
