# pytest function to test read and load data script
import pytest 

url="https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv"
url2 = ...
url3 = 123
url4 = "  "

def validate_url(url):
    """
    Validation tests copied from the read_trump_tweets.py script
    """
    if url is None:
        raise ValueError("URL cannot be None")
    if not isinstance(url, str):
        raise TypeError(f"URL must be a string, got {type(url).__name__}")
    if not url.strip():
       raise ValueError("URL cannot be empty")
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    return url 


def test_url_validation():
    """
    Pytest function to test the url validation methods in the script
    """

    with pytest.raises(ValueError, match="URL cannot be None"):
        validate_url(None) 

    with pytest.raises(TypeError, match="URL must be a string"):
        validate_url(123) 

    with pytest.raises(ValueError, match="URL cannot be empty"):
        validate_url("") 

    with pytest.raises(ValueError, match="URL cannot be empty"):
        validate_url("  ") 

    with pytest.raises(ValueError, match="URL must start with http:// or https://"):
        validate_url("ftp://example.com") 

    with pytest.raises(ValueError, match="URL cannot be empty"):
        validate_url("") 


if __name__ == "__main__":
    test_url_validation()
    print("All tests passed!")
    

    
    
    