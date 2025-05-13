python -m pyserini.index.lucene -collection JsonCollection \
-input ./document_jsons_final -index ./BM25index_final \
-generator DefaultLuceneDocumentGenerator -threads 16 -storePositions -storeDocvectors -storeRaw