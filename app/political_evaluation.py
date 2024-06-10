import llmAccess as llm 
import requests
import re
from bs4 import BeautifulSoup

def extract_text_from_url(url, identifiers, is_class=True):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Create an empty list to store the extracted text
        extracted_text = []
        
        # Function to recursively extract text from elements
        def extract_text_recursive(element):
            # If the element is a string, append its text to the list
            if isinstance(element, str):
                extracted_text.append(element.strip())
            # If the element is a tag, recurse on its children
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
        
        # Combine the extracted text
        combined_text = "\n".join(extracted_text)
        
        return combined_text
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
                'url': 'https://www.miningweekly.com/page/manganese',
                'identifiers': ['card-title', 'cm-last-updated'],
                'is_class': True
            },
            {
                'url': 'https://www.northernminer.com/commodity/manganese/',
                'identifiers': ['entry-meta-date', 'content-list-title'],
                'is_class': True
            }
        ],
        'molybdenum': [
            {
                'url': 'https://www.northernminer.com/commodity/molybdenum/',
                'identifiers': ['main-content'],
                'is_class': False
            }
        ]
    }
    
    score = 0.0
    sources_asked = 0
    for site in news_sites[mineral]:
        prompt = ""
        prompt += str(extract_text_from_url(site['url'], site['identifiers'], site['is_class']))
        prompt += "  - Rate how these news in total affect manganese prices from -1.0 as prices drop to 1.0 as prices rise. Ignore news not between " + date_from + " to " + date_to + ". If none matches, rate only news with closest date. give as response only a number, don't write anything else."
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" + prompt)
        answear = llm.send_prompt(prompt)
        #print(answear)
        try:
            value = extract_first_float(answear)
            score += value
            sources_asked += 1
        except ValueError: # this MF meta AI can't follow instructions, abandon ship!
            pass
    
    if sources_asked == 0:
        return 0.0
    else:
        return score/sources_asked