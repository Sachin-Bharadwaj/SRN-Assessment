import os
import time
import glob
import json
import requests
import pandas as pd
import logging
from openai import OpenAI
from dotenv import load_dotenv
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"linkedin_api_extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Configuration - PhantomBuster API
PHANTOMBUSTER_API_KEY = os.getenv("PHANTOMBUSTER_API_KEY")
PHANTOMBUSTER_LINKEDIN_SCRAPER_ID = os.getenv("PHANTOMBUSTER_LINKEDIN_SCRAPER_ID")

# Configuration - OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "5"))  # Default 5 seconds between requests
MAX_PROFILES_PER_RUN = int(os.getenv("MAX_PROFILES_PER_RUN", "250"))  # Default max profiles per run

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def launch_phantombuster_scrape(linkedin_url):
    """
    Launch PhantomBuster API to scrape a LinkedIn profile
    
    Args:
        linkedin_url: LinkedIn profile URL to scrape
        
    Returns:
        Container ID if successful, None otherwise
    """
    try:
        logging.info(f"Launching PhantomBuster scrape for: {linkedin_url}")
        
        # PhantomBuster launch API endpoint
        url = f"https://api.phantombuster.com/api/v2/agents/{PHANTOMBUSTER_LINKEDIN_SCRAPER_ID}/launch"
        
        # Request headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Phantombuster-Key": PHANTOMBUSTER_API_KEY
        }
        
        # Request body - Configure the LinkedIn Profile Scraper agent
        data = {
            "argument": json.dumps({
                "sessionCookie": os.getenv("LINKEDIN_SESSION_COOKIE", ""),
                "profileUrls": linkedin_url,
                "extractDefaultUrl": True
            })
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            container_id = response.json().get("containerId")
            logging.info(f"PhantomBuster scrape launched successfully. Container ID: {container_id}")
            return container_id
        else:
            logging.error(f"Failed to launch PhantomBuster scrape. Status Code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error launching PhantomBuster scrape: {str(e)}")
        return None

def check_phantombuster_status(container_id):
    """
    Check the status of a PhantomBuster container
    
    Args:
        container_id: PhantomBuster container ID
        
    Returns:
        True if completed, False otherwise
    """
    try:
        logging.info(f"Checking PhantomBuster container status: {container_id}")
        
        # PhantomBuster status API endpoint
        url = f"https://api.phantombuster.com/api/v2/containers/{container_id}"
        
        # Request headers
        headers = {
            "Accept": "application/json",
            "X-Phantombuster-Key": PHANTOMBUSTER_API_KEY
        }
        
        # Make the API request
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            status = response.json().get("status")
            logging.info(f"PhantomBuster container status: {status}")
            
            # Check if the container has completed
            if status in ["finished", "stopped"]:
                return True
            else:
                return False
        else:
            logging.error(f"Failed to check PhantomBuster status. Status Code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error checking PhantomBuster status: {str(e)}")
        return False

def fetch_phantombuster_results(container_id):
    """
    Fetch the results from a PhantomBuster container
    
    Args:
        container_id: PhantomBuster container ID
        
    Returns:
        Profile data if successful, None otherwise
    """
    try:
        logging.info(f"Fetching PhantomBuster results for container: {container_id}")
        
        # PhantomBuster output API endpoint
        url = f"https://api.phantombuster.com/api/v2/containers/{container_id}/output"
        
        # Request headers
        headers = {
            "Accept": "application/json",
            "X-Phantombuster-Key": PHANTOMBUSTER_API_KEY
        }
        
        # Make the API request
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            output = response.json()
            
            # Check if there's an output URL (contains the result JSON)
            output_url = output.get("outputUrl")
            
            if output_url:
                # Fetch the JSON result
                result_response = requests.get(output_url)
                
                if result_response.status_code == 200:
                    try:
                        profile_data = result_response.json()
                        logging.info(f"Successfully fetched profile data")
                        return profile_data
                    except Exception as e:
                        logging.error(f"Error parsing profile data: {str(e)}")
                        return None
                else:
                    logging.error(f"Failed to fetch profile data. Status Code: {result_response.status_code}")
                    return None
            else:
                logging.error("No output URL found in PhantomBuster response")
                return None
        else:
            logging.error(f"Failed to fetch PhantomBuster results. Status Code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching PhantomBuster results: {str(e)}")
        return None

def analyze_with_gpt4o(profile_data, linkedin_url):
    """
    Analyze LinkedIn profile data using GPT-4o
    
    Args:
        profile_data: Profile data from PhantomBuster
        linkedin_url: LinkedIn profile URL
        
    Returns:
        Dictionary with extracted information
    """
    try:
        logging.info(f"Analyzing profile with GPT-4o: {linkedin_url}")
        
        # Convert profile data to a readable format for GPT-4o
        profile_summary = json.dumps(profile_data, indent=2)
        
        prompt = f"""
        You are an expert at analyzing LinkedIn profiles for talent acquisition.
        
        Based on the following LinkedIn profile data, extract:
        
        1. Title: The person's current job title
        2. Tenure: How long they've been at their current job
        3. Startup Fit: Assess on a scale of 1-10 how well they might fit at a startup based on their experience
        4. Tech Stack: List the main technologies and tools they appear to be familiar with
        
        Return ONLY a JSON object with these 4 fields. If you can't determine any field, use null or "Unknown".
        
        LinkedIn Profile Data:
        {profile_summary[:7000]}  # Truncated to avoid token limits
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

def process_profile_with_api(linkedin_url):
    """
    Process a LinkedIn profile using PhantomBuster API
    
    Args:
        linkedin_url: LinkedIn profile URL
        
    Returns:
        Dictionary with extracted information or None if failed
    """
    try:
        # Launch PhantomBuster scrape
        container_id = launch_phantombuster_scrape(linkedin_url)
        
        if not container_id:
            logging.error(f"Failed to launch PhantomBuster scrape for {linkedin_url}")
            return None
        
        # Wait for the scrape to complete (with timeout)
        max_attempts = 10
        attempt = 0
        completed = False
        
        while attempt < max_attempts and not completed:
            attempt += 1
            
            # Wait before checking status
            time.sleep(10)  # Wait 10 seconds between status checks
            
            # Check status
            completed = check_phantombuster_status(container_id)
            
            if completed:
                logging.info(f"PhantomBuster scrape completed for {linkedin_url}")
                break
            
            logging.info(f"Waiting for PhantomBuster scrape to complete... Attempt {attempt}/{max_attempts}")
        
        if not completed:
            logging.error(f"Timed out waiting for PhantomBuster scrape to complete for {linkedin_url}")
            return None
        
        # Fetch results
        profile_data = fetch_phantombuster_results(container_id)
        
        if not profile_data:
            logging.error(f"Failed to fetch profile data for {linkedin_url}")
            return None
        
        # Analyze with GPT-4o
        analysis_result = analyze_with_gpt4o(profile_data, linkedin_url)
        
        if not analysis_result:
            logging.error(f"Failed to analyze profile data with GPT-4o for {linkedin_url}")
            return None
        
        # Parse the analysis result
        try:
            analysis_data = json.loads(analysis_result)
            return analysis_data
        except Exception as e:
            logging.error(f"Error parsing analysis result: {str(e)}")
            return None
        
    except Exception as e:
        logging.error(f"Error processing profile with API: {str(e)}")
        return None

def fallback_analysis_from_csv_data(row):
    """
    Fallback method to analyze profile when API fails, using data from the CSV
    
    Args:
        row: DataFrame row with profile data
        
    Returns:
        Dictionary with extracted information or None if failed
    """
    try:
        logging.info(f"Using fallback analysis for profile: {row['LinkedIn']}")
        
        # Create a prompt with the data we have from the CSV
        prompt = f"""
        You are an expert at analyzing professional profiles for talent acquisition.
        
        Based on the following limited profile information, extract and infer:
        
        1. Title: The person's current job title
        2. Tenure: How long they've been at their current job (if unavailable, indicate "Unknown")
        3. Startup Fit: Assess on a scale of 1-10 how well they might fit at a startup based on their experience
        4. Tech Stack: List the main technologies and tools they might be familiar with based on their role and company
        
        Limited Profile Information:
        - Name: {row.get('First name', '')} {row.get('Last name', '')}
        - LinkedIn URL: {row['LinkedIn']}
        - Current Title: {row.get('Current Title', 'Unknown')}
        - Current Organization: {row.get('Current Org Name', 'Unknown')}
        - Education: {row.get('Education', 'Unknown')}
        - Location: {row.get('Location', 'Unknown')}
        
        Return ONLY a JSON object with these 4 fields (Title, Tenure, Startup Fit, Tech Stack).
        If you can't determine any field, use null or "Unknown".
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Parse the analysis result
        try:
            analysis_data = json.loads(result)
            return analysis_data
        except Exception as e:
            logging.error(f"Error parsing fallback analysis result: {str(e)}")
            return None
        
    except Exception as e:
        logging.error(f"Error in fallback analysis: {str(e)}")
        return None

def process_csv_file(csv_file, max_profiles=MAX_PROFILES_PER_RUN, use_api=True):
    """
    Process a single CSV file
    
    Args:
        csv_file: Path to the CSV file
        max_profiles: Maximum number of profiles to process in a single run
        use_api: Whether to use PhantomBuster API (if False, use fallback)
        
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
        
        # Count profiles to process (not yet processed)
        profiles_to_process = df[df['Title_GPT'].isna() & df['LinkedIn'].notna() & 
                                df['LinkedIn'].str.startswith('https://linkedin.com/', na=False)]
        
        logging.info(f"Found {len(profiles_to_process)} profiles to process")
        
        # Limit to max_profiles
        profiles_to_process = profiles_to_process.head(max_profiles)
        logging.info(f"Processing {len(profiles_to_process)} profiles in this run")
        
        # Process each row
        processed_count = 0
        for idx, row in profiles_to_process.iterrows():
            linkedin_url = row['LinkedIn']
            
            logging.info(f"Processing profile {processed_count+1}/{len(profiles_to_process)}: {linkedin_url}")
            
            analysis_data = None
            
            if use_api and PHANTOMBUSTER_API_KEY and PHANTOMBUSTER_LINKEDIN_SCRAPER_ID:
                # Use PhantomBuster API
                analysis_data = process_profile_with_api(linkedin_url)
                
                if not analysis_data:
                    logging.warning(f"API processing failed, falling back to CSV data analysis")
                    analysis_data = fallback_analysis_from_csv_data(row)
            else:
                # Use fallback method
                logging.info(f"Using fallback method (no API credentials or API disabled)")
                analysis_data = fallback_analysis_from_csv_data(row)
            
            if analysis_data:
                # Update DataFrame
                df.loc[idx, 'Title_GPT'] = analysis_data.get('Title', 'Unknown')
                df.loc[idx, 'Tenure_GPT'] = analysis_data.get('Tenure', 'Unknown')
                df.loc[idx, 'StartupFit_GPT'] = analysis_data.get('Startup Fit', 'Unknown')

                # Fix for TechStack - convert to string if it's a list/iterable
                tech_stack = analysis_data.get('Tech Stack', 'Unknown')
                if isinstance(tech_stack, (list, tuple)):
                    tech_stack = ', '.join(tech_stack)
                df.loc[idx, 'TechStack_GPT'] = tech_stack
                
                
                # Save changes after each successful analysis to prevent data loss
                df.to_csv(csv_file, index=False)
                logging.info(f"Updated and saved profile data for {linkedin_url}")
                
                processed_count += 1
            
            # Add delay to avoid rate limits
            delay = REQUEST_DELAY + random.uniform(0, 2)
            time.sleep(delay)
        
        # Final save
        df.to_csv(csv_file, index=False)
        logging.info(f"Completed processing file: {csv_file}. Processed {processed_count} profiles.")
    
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file}: {str(e)}")

def main():
    """Main function to extract LinkedIn profile information using API"""
    logging.info("Starting LinkedIn profile information extraction using API")
    
    # Check if API keys are set
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    use_api = True
    if not PHANTOMBUSTER_API_KEY or not PHANTOMBUSTER_LINKEDIN_SCRAPER_ID:
        logging.warning("PhantomBuster API credentials not found. Will use fallback method.")
        use_api = False
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob('./data/SRN_AI_Engineer_Assessment/JuiceboxExport_*.csv')
    
    if not csv_files:
        logging.error("No CSV files found matching the pattern")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, use_api=use_api)
    
    logging.info("LinkedIn profile information extraction completed")

if __name__ == "__main__":
    main() 