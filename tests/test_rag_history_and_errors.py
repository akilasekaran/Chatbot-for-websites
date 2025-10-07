import unittest
from unittest import mock

import sys
sys.path.insert(0, '')

from web_scraper_rag import (
    build_rag_chain,
    build_retriever,
    run_query,
)

class DummyRetriever:
    def __init__(self, docs=None):
        self._docs = docs or []

    def get_relevant_documents(self, q):
        return self._docs

class DummyDocs:
    def __init__(self, text):
        self.page_content = text

class DummyCombineChain:
    def invoke(self, inputs):
        # expects context key to be present
        ctx = inputs.get('context', [])
        if not ctx:
            return {'answer': 'no context'}
        return {'answer': 'got:' + ctx[0].page_content}

class TestHistoryAndErrors(unittest.TestCase):
    def test_history_aware_chain_with_messages(self):
        # Simulate docs returned and ensure history-aware chain can be built
        dummy_llm = mock.Mock()
        docs = [DummyDocs('history doc')]
        retriever = DummyRetriever(docs=docs)

        # build history-aware chain, should not raise
        chain = build_rag_chain(dummy_llm, retriever, use_history=True)

        # Patch chain.invoke to return a predictable value if needed
        with mock.patch.object(chain, 'invoke', return_value={'answer': 'ok-history'}):
            resp = run_query(chain, 'ask')
            self.assertEqual(resp, 'ok-history')

    def test_run_query_handles_chain_exceptions(self):
        class BrokenChain:
            def invoke(self, inp):
                raise RuntimeError('boom')

        broken = BrokenChain()
        # run_query should propagate the exception (test documents expected behavior)
        with self.assertRaises(RuntimeError):
            run_query(broken, 'x')

    @mock.patch('web_scraper_rag.WebBaseLoader')
    @mock.patch('web_scraper_rag.Chroma')
    @mock.patch('web_scraper_rag.OpenAIEmbeddings')
    def test_build_retriever_loader_error(self, mock_embeddings, mock_chroma, mock_loader):
        # Simulate loader.load raising an exception
        mock_loader.return_value.load.side_effect = Exception('network fail')
        # The function should bubble the exception up — or at least not crash with unrelated errors
        with self.assertRaises(Exception):
            _ = build_retriever(['https://example.com'])

if __name__ == '__main__':
    unittest.main()
