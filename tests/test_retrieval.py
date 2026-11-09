from src.retrieval import PolicyRetriever

def test_retriever_finds_income_policy():
    retriever = PolicyRetriever()
    results = retriever.search("How can an applicant verify income?")
    assert results
    assert any("income" in item["text"].lower() for item in results)
