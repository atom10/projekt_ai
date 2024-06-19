import llmAccess as llm 
import requests
import re
from bs4 import BeautifulSoup
import feedparser

cache = []

def extract_text_from_url(url, identifiers, is_class=True):
    # Define headers for making the request
    request_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=request_headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Check if the response is HTML or RSS/XML
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'xml' in content_type or 'rss' in content_type:
            # Parse RSS/XML feed
            feed = feedparser.parse(response.text)
            
            # Extract text from RSS items
            extracted_text = []
            for entry in feed.entries:
                if 'summary' in entry:
                    extracted_text.append(entry.summary)
                elif 'description' in entry:
                    extracted_text.append(entry.description)
                elif 'title' in entry:
                    extracted_text.append(entry.title)
                
                # Check for various possible date attributes
                pub_date = entry.get('pubDate') or entry.get('published') or entry.get('updated')
                if pub_date:
                    extracted_text.append(f"Published on: {pub_date}")
                    
            combined_text = "\n".join(extracted_text)
        
        else:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Create an empty list to store the extracted text
            extracted_text = []
            
            # Function to recursively extract text from elements
            def extract_text_recursive(element):
                if isinstance(element, str):
                    extracted_text.append(element.strip())
                elif hasattr(element, 'children'):
                    for child in element.children:
                        extract_text_recursive(child)
            
            # Find all elements based on class name or ID
            if is_class:
                elements = soup.find_all(class_=identifiers)
            else:
                elements = soup.find_all(id=identifiers)
            
            # Extract text from found elements
            for element in elements:
                extract_text_recursive(element)
            
            combined_text = "\n".join(extracted_text)
        
        return combined_text.strip()
    
    else:
        # Request was not successful
        print("Failed to retrieve content from the URL.")
        print(str(response))
        return None

def extract_first_float(line):
    # Use a regular expression to find all potential float values in the string
    float_pattern = re.compile(r'-?\d+\.\d+')
    matches = float_pattern.findall(line)
    
    for match in matches:
        try:
            # Try converting each match to a float
            return float(match)
        except ValueError:
            # If conversion fails, continue to the next match
            continue
    
    # If no valid float value is found, return 0.0
    return 0.0

def evaluate(mineral, date_from, date_to):
    news_sites = {
        'manganese': [
            {
                'url': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            },
            {
                'url': 'https://feeds.bbci.co.uk/news/rss.xml?edition=us',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            },
            {
                'url': 'https://feeds.bbci.co.uk/news/rss.xml?edition=int',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            }
        ],
        'molybdenum': [
            {
                'url': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            },
            {
                'url': 'https://feeds.bbci.co.uk/news/rss.xml?edition=us',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            },
            {
                'url': 'https://feeds.bbci.co.uk/news/rss.xml?edition=int',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            }
        ]
    }
    
    score = 0.0
    sources_asked = 0
    for site in news_sites[mineral]:
        prompt = ""
        for cached_url, cached_prompt in cache:
            if cached_url == site['url']:
                prompt = cached_prompt
                break
        if prompt == "":
            prompt += str(extract_text_from_url(site['url'], site['identifiers'], site['is_class']))
            prompt += "  - Rate how these news in total affect manganese prices from -1.0 as prices drop to 1.0 as prices rise. Ignore news not between " + date_from + " to " + date_to + ". If none matches, rate only news with closest date. give as response only a number, don't write anything else."
            cache.append((site['url'], prompt))
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" + prompt)
        answear = llm.send_prompt(prompt)
        #print(answear)
        if answear is not None:
            try:
                value = extract_first_float(answear)
                score += value
                sources_asked += 1
            except ValueError: # this MF meta AI can't follow instructions, abandon ship!
                pass
    
    if sources_asked == 0:
        return [0.0]
    else:
        return [score/sources_asked]