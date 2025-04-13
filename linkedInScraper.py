import os
import time
import glob
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from fake_useragent import UserAgent
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("linkedin_scraper.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set your proxies list - can be loaded from a file or environment variable
PROXIES = os.getenv("PROXIES", "").split(",") if os.getenv("PROXIES") else []
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2"))  # Default 2 seconds between requests
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_random_proxy():
    """Get a random proxy from the list, or None if no proxies available"""
    if not PROXIES:
        return None
    return random.choice(PROXIES)

def get_random_user_agent():
    """Generate a random user agent"""
    ua = UserAgent()
    return ua.random

def extract_linkedin_profile(url, max_retries=3):
    """
    Extract content from a LinkedIn profile
    
    Args:
        url: LinkedIn profile URL
        max_retries: Maximum number of retries
        
    Returns:
        HTML content of the profile page or None if failed
    """
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    proxy = get_random_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Fetching profile: {url} (Attempt {attempt+1}/{max_retries})")
            
            response = requests.get(
                url, 
                headers=headers, 
                proxies=proxies,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.text
            
            logging.warning(f"Failed to fetch profile: {url} - Status Code: {response.status_code}")
            
            # Incremental backoff
            time.sleep((attempt + 1) * REQUEST_DELAY)
            
            # Get a new proxy for the next attempt
            proxy = get_random_proxy()
            proxies = {"http": proxy, "https": proxy} if proxy else None
            
        except Exception as e:
            logging.error(f"Error fetching profile {url}: {str(e)}")
            time.sleep((attempt + 1) * REQUEST_DELAY)
    
    logging.error(f"Max retries exceeded for {url}")
    return None

def analyze_with_gpt4o(profile_content, linkedin_url):
    """
    Analyze LinkedIn profile content using GPT-4o
    
    Args:
        profile_content: HTML content of the profile
        linkedin_url: Original LinkedIn URL for reference
        
    Returns:
        Dictionary with extracted information
    """
    try:
        logging.info(f"Analyzing profile with GPT-4o: {linkedin_url}")
        
        prompt = f"""
        You are an expert at analyzing LinkedIn profiles. Extract the following information from this profile:
        
        1. Title: The person's current job title
        2. Tenure: How long they've been at their current job
        3. Startup Fit: Assess on a scale of 1-10 how well they might fit at a startup based on their experience
        4. Tech Stack: List the main technologies and tools they appear to be familiar with
        
        Return ONLY a JSON object with these 4 fields. If you can't determine any field, use null or "Unknown".
        
        Here's the LinkedIn profile content to analyze:
        
        {profile_content[:10000]}  # Truncate to avoid token limits
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        return result
    
    except Exception as e:
        logging.error(f"Error analyzing profile with GPT-4o: {str(e)}")
        return None

def process_csv_file(csv_file):
    """
    Process a single CSV file
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        None
    """
    try:
        logging.info(f"Processing file: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Create new columns if they don't exist
        for col in ['Title_GPT', 'Tenure_GPT', 'StartupFit_GPT', 'TechStack_GPT']:
            if col not in df.columns:
                df[col] = None
        
        # Get list of LinkedIn URLs
        linkedin_urls = df['LinkedIn'].tolist()
        
        # Process each LinkedIn URL
        for idx, url in enumerate(linkedin_urls):
            # Skip if not a valid URL
            if not isinstance(url, str) or not url.startswith('https://linkedin.com/'):
                continue
                
            # Skip if already processed
            if not pd.isna(df.loc[idx, 'Title_GPT']):
                logging.info(f"Skipping already processed profile: {url}")
                continue
                
            logging.info(f"Processing profile {idx+1}/{len(linkedin_urls)}: {url}")
            
            # Extract profile content
            profile_content = extract_linkedin_profile(url)
            
            if profile_content:
                # Analyze with GPT-4o
                analysis_result = analyze_with_gpt4o(profile_content, url)
                
                if analysis_result:
                    try:
                        import json
                        analysis_data = json.loads(analysis_result)
                        
                        # Update DataFrame
                        df.loc[idx, 'Title_GPT'] = analysis_data.get('Title', 'Unknown')
                        df.loc[idx, 'Tenure_GPT'] = analysis_data.get('Tenure', 'Unknown')
                        df.loc[idx, 'StartupFit_GPT'] = analysis_data.get('Startup Fit', 'Unknown')
                        df.loc[idx, 'TechStack_GPT'] = analysis_data.get('Tech Stack', 'Unknown')
                        
                        # Save changes after each successful analysis to prevent data loss
                        df.to_csv(csv_file, index=False)
                        logging.info(f"Updated and saved profile data for {url}")
                    
                    except Exception as e:
                        logging.error(f"Error processing analysis result: {str(e)}")
            
            # Add delay to avoid rate limits
            time.sleep(REQUEST_DELAY)
        
        # Final save
        df.to_csv(csv_file, index=False)
        logging.info(f"Completed processing file: {csv_file}")
    
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file}: {str(e)}")

def main():
    """Main function to run the LinkedIn scraper"""
    logging.info("Starting LinkedIn profile scraper")
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob('./data/SRN_AI_Engineer_Assessment/JuiceboxExport_*.csv')
    
    if not csv_files:
        logging.error("No CSV files found matching the pattern")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file)
    
    logging.info("LinkedIn profile scraping completed")

if __name__ == "__main__":
    main()
