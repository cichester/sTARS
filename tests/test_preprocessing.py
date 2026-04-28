import polars as pl
from src.preprocessing import clean_text_columns

def test_clean_text_columns_removes_html_and_urls():
    df = pl.DataFrame({
        "title": ["Hello <br> world!", "Great http://example.com"],
        "text": ["This is an <a href='link'>awesome</a> product.", "Check this out: www.test.org   "]
    })
    
    cleaned_df = clean_text_columns(df)
    
    # Titles should be clean
    assert cleaned_df["title"].to_list()[0] == "Hello world!"
    assert cleaned_df["title"].to_list()[1] == "Great"
    
    # Texts should be clean
    assert cleaned_df["text"].to_list()[0] == "This is an awesome product."
    assert cleaned_df["text"].to_list()[1] == "Check this out:"
    
    # review_text should be concatenated correctly
    assert cleaned_df["review_text"].to_list()[0] == "Hello world! This is an awesome product."
    assert cleaned_df["review_text"].to_list()[1] == "Great Check this out:"

def test_clean_text_columns_handles_nulls():
    df = pl.DataFrame({
        "title": ["Valid title", None],
        "text": [None, "Valid text"]
    })
    
    cleaned_df = clean_text_columns(df)
    
    # Nulls should be replaced with empty str
    assert cleaned_df["review_text"].to_list()[0] == "Valid title"
    assert cleaned_df["review_text"].to_list()[1] == "Valid text"
