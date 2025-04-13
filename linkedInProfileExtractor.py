import os
import time
import glob
import json
import pandas as pd
import logging
from openai import OpenAI
from dotenv import load_dotenv
import random
from tqdm import tqdm


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("linkedin_extractor.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1"))  # Default 1 second between requests
MAX_PROFILES_PER_RUN = int(os.getenv("MAX_PROFILES_PER_RUN", "250"))  # Process limited profiles per run

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def simulate_profile_info_from_url(linkedin_url, person_name, current_job, education):
    """
    Instead of actual scraping, simulate profile information based on provided info
    This avoids LinkedIn scraping restrictions
    
    Args:
        linkedin_url: LinkedIn profile URL
        person_name: Person's name from CSV
        current_job: Current job information
        education: Education information
        
    Returns:
        Simulated profile information
    """
    # Extract username from LinkedIn URL
    username = linkedin_url.split('/in/')[-1] if '/in/' in linkedin_url else ""
    
    # Create a simulated profile based on available information
    profile_info = {
        "url": linkedin_url,
        "username": username,
        "name": person_name,
        "current_position": current_job,
        "education": education
    }
    
    return profile_info

def extract_info_with_gpt4o(profile_info, linkedin_url):
    """
    Use GPT-4o to extract information about a person from their LinkedIn profile data
    
    Args:
        profile_info: Dictionary containing profile information
        linkedin_url: LinkedIn profile URL
        
    Returns:
        Dictionary with extracted information
    """
    try:
        logging.info(f"Analyzing profile with GPT-4o: {linkedin_url}")
        
        prompt = f"""
        You are an expert at analyzing professional profiles and extracting relevant information.
        
        Based on the following limited profile information, I need you to extract and infer:
        
        1. Title: The person's current job title
        2. Tenure: How long they've been at their current job (if available, otherwise indicate unknown)
        3. Startup Fit: Assess on a scale of 1-10 how well they might fit at a startup based on their experience and background
        4. Tech Stack: List the main technologies and tools they are likely familiar with based on their role and company
        
        LinkedIn URL: {linkedin_url}
        Name: {profile_info.get('name', 'Unknown')}
        Current Position: {profile_info.get('current_position', 'Unknown')}
        Education: {profile_info.get('education', 'Unknown')}
        
        Return ONLY a JSON object with these 4 fields. If you can't determine any field, use null or "Unknown".
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

def process_csv_file(csv_file, max_profiles=MAX_PROFILES_PER_RUN):
    """
    Process a single CSV file
    
    Args:
        csv_file: Path to the CSV file
        max_profiles: Maximum number of profiles to process in a single run
        
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
        
        # Add tqdm progress bar
        for idx, row in tqdm(profiles_to_process.iterrows(), total=len(profiles_to_process), 
                             desc="Processing LinkedIn Profiles", unit="profile"):
            linkedin_url = row['LinkedIn']
            first_name = row.get('First name', '')
            last_name = row.get('Last name', '')
            full_name = f"{first_name} {last_name}".strip()
            current_job = f"{row.get('Current Title', 'Unknown')} at {row.get('Current Org Name', 'Unknown')}"
            education = row.get('Education', 'Unknown')
            
            logging.info(f"Processing profile {processed_count+1}/{len(profiles_to_process)}: {linkedin_url}")
            
            # Simulate profile information instead of scraping
            profile_info = simulate_profile_info_from_url(linkedin_url, full_name, current_job, education)
            
            # Add some randomness to the delay (1-3 seconds)
            delay = REQUEST_DELAY + random.uniform(0, 2)
            
            # Extract information with GPT-4o
            analysis_result = extract_info_with_gpt4o(profile_info, linkedin_url)
            
            if analysis_result:
                try:
                    analysis_data = json.loads(analysis_result)
                    
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
                
                except Exception as e:
                    logging.error(f"Error processing analysis result: {str(e)}")
            
            # Add delay to avoid rate limits
            time.sleep(delay)
        
        # Final save
        df.to_csv(csv_file, index=False)
        logging.info(f"Completed processing file: {csv_file}. Processed {processed_count} profiles.")
    
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file}: {str(e)}")

def main():
    """Main function to extract LinkedIn profile information"""
    logging.info("Starting LinkedIn profile information extraction")
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob('./data/SRN_AI_Engineer_Assessment/JuiceboxExport_*.csv')
    
    if not csv_files:
        logging.error("No CSV files found matching the pattern")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file with tqdm progress bar
    for csv_file in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        process_csv_file(csv_file)
    
    logging.info("LinkedIn profile information extraction completed")

if __name__ == "__main__":
    main() 