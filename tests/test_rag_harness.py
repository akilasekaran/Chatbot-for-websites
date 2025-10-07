import unittest
from unittest import mock

import importlib
import sys

# Make sure project root is on path
sys.path.insert(0, '')

from web_scraper_rag import (
    ensure_openai_key,
    build_retriever,
    build_rag_chain,
    run_query,
)

class DummyRetriever:
    def get_relevant_documents(self, q):
        class D:
            page_content = 'dummy content about task decomposition'
        return [D()]

class DummyChain:
    def invoke(self, inp):
        return {'answer': 'dummy answer'}

class TestRAGHarness(unittest.TestCase):
    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_build_retriever_smoke(self, mock_embeddings, mock_chroma, mock_loader):
        # loader.load should return a list-like of documents
        mock_loader.return_value.load.return_value = [mock.Mock(page_content='x')]
        mock_chroma.from_documents.return_value.as_retriever.return_value = DummyRetriever()

        retriever = build_retriever(['https://example.com'])
        docs = retriever.get_relevant_documents('q')
        self.assertTrue(len(docs) > 0)
        self.assertIn('task decomposition', docs[0].page_content)

    def test_build_rag_chain_and_run_query(self):
        # Create a dummy llm and retriever
        dummy_llm = mock.Mock()
        dummy_retriever = DummyRetriever()
        # Build a simple chain using the real function but inject dummy components
        chain = build_rag_chain(dummy_llm, dummy_retriever, use_history=False)
        # The real function returns a chain object; for safety, we'll not rely on internal type
        # Instead, patch run_query to accept the dummy chain
        with mock.patch.object(chain, 'invoke', return_value={'answer': 'ok'}):
            ans = run_query(chain, 'hello')
            self.assertEqual(ans, 'ok')

if __name__ == '__main__':
    unittest.main()
