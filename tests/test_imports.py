from scripts.custom_transformers import TextStatsTransformer


def test_custom_transformer_constructs():
    t = TextStatsTransformer(inputCol="review_text")
    assert t.getOrDefault(t.inputCol) == "review_text"
