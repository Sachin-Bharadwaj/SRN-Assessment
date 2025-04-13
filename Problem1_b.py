from dotenv import load_dotenv
import os
import PyPDF2
import json
import glob
from openai import OpenAI
import logging
from tqdm import tqdm
import time
import re
import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Make sure it's set in the .env file.")

client = OpenAI(api_key=openai_api_key)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean and format text for GPT analysis."""
    if not text:
        return ""
    
    # Remove special characters and normalize whitespace
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = ' '.join(text.split())
    
    # Remove any remaining non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
    
    return text.strip()

def load_job_descriptions(json_path):
    """Load and process job descriptions from JSON file."""
    try:
        with open(json_path, 'r') as file:
            raw_data = json.load(file)
            
        # Process the job descriptions into a more usable format
        processed_jobs = []
        for company, details in raw_data.items():
            # Extract the job description from the details array
            job_desc = None
            for detail in details:
                if isinstance(detail, str) and "Extracted from page\n:" in detail:
                    try:
                        # Try to parse the JSON string from the detail
                        json_str = detail.split("```json\n")[1].split("\n```")[0]
                        job_data = json.loads(json_str)
                        job_desc = job_data
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON for {company}: {e}")
                        continue
            
            if job_desc:
                # Create a structured job description
                description = ""
                if isinstance(job_desc, dict):
                    # Extract relevant fields for the description
                    if 'Job Description' in job_desc:
                        if isinstance(job_desc['Job Description'], dict):
                            # Handle nested job description
                            desc_parts = []
                            for key, value in job_desc['Job Description'].items():
                                if isinstance(value, list):
                                    desc_parts.append(f"{key}:\n" + "\n".join(f"- {item}" for item in value))
                                else:
                                    desc_parts.append(f"{key}: {value}")
                            description = "\n\n".join(desc_parts)
                        else:
                            description = job_desc['Job Description']
                    else:
                        # If no Job Description field, use other relevant fields
                        desc_parts = []
                        for key in ['About the Company', 'Roles and Responsibilities', 'Job Requirements']:
                            if key in job_desc:
                                if isinstance(job_desc[key], list):
                                    desc_parts.append(f"{key}:\n" + "\n".join(f"- {item}" for item in job_desc[key]))
                                else:
                                    desc_parts.append(f"{key}: {job_desc[key]}")
                        description = "\n\n".join(desc_parts)
                
                processed_job = {
                    'company': company.strip(),
                    'description': clean_text(description),
                    'tech_stack': job_desc.get('Tech Stack', []) if isinstance(job_desc, dict) else [],
                    'role': job_desc.get('Role', '') if isinstance(job_desc, dict) else '',
                    'locations': job_desc.get('Locations', '') if isinstance(job_desc, dict) else ''
                }
                processed_jobs.append(processed_job)
        
        return processed_jobs
    except Exception as e:
        logger.error(f"Error loading job descriptions from {json_path}: {e}")
        return []

def compare_resume_with_job(resume_text, job_description, use_api=True):
    """
    Compare a resume with a job description and return a match score with justification.
    Can use either OpenAI completion API or embeddings for comparison.
    """
    if use_api:
        # Using OpenAI Chat API for more detailed analysis
        try:
            prompt = f"""
            Task: Evaluate how well a candidate's resume matches a job description.
            
            Instructions:
            1. Analyze the skills, experience, education, and qualifications in both the resume and job description
            2. Consider both direct matches and transferable skills
            3. Provide a match score between 1 and 10 (with 10 being the highest match)
            4. Provide a brief justification for your score (2-3 sentences max)
            
            Your response should be in this format:
            Score: [number between 1-10]
            Justification: [brief explanation]
            
            Resume:
            {resume_text[:3500]}  # Limiting resume text to avoid token limits
            
            Job Description:
            {job_description['description'][:3500]}  # Using the correct field name
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert HR recruiter who analyzes resumes and job descriptions to determine match percentages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01,
                max_tokens=250
            )
            
            # Extract the match percentage and justification from the response
            match_text = response.choices[0].message.content.strip()
            
            # Extract score (pattern: "Score: X" or just a number)
            score_match = re.search(r'Score:\s*(\d+)', match_text, re.IGNORECASE)
            if not score_match:
                score_match = re.search(r'(\d+)\s*/\s*10', match_text)  # Look for "X/10" format
            if not score_match:
                score_match = re.search(r'(\d+)', match_text)  # Just find any number
                
            # Extract justification (pattern: "Justification: [text]")
            justification_match = re.search(r'Justification:\s*(.*)', match_text, re.IGNORECASE | re.DOTALL)
            
            # Set defaults if not found
            score = 5  # Default score
            justification = "No justification provided."
            
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is between 1-10
                score = min(max(score, 1), 10)
                
            if justification_match:
                justification = justification_match.group(1).strip()
            else:
                # If no explicit justification section, use everything after the score as justification
                score_index = match_text.find(str(int(score)))
                if score_index != -1 and score_index + len(str(int(score))) < len(match_text):
                    remaining_text = match_text[score_index + len(str(int(score))):].strip()
                    # Remove any non-letter starting chars like ":" or "/"
                    remaining_text = re.sub(r'^[^a-zA-Z]*', '', remaining_text)
                    if remaining_text:
                        justification = remaining_text
            
            # Convert to percentage format for consistency
            return score, justification
                
        except Exception as e:
            logger.error(f"Error using OpenAI API: {e}")
            
            # Implement exponential backoff for rate limits
            if "rate limit" in str(e).lower():
                wait_time = 5
                logger.info(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return compare_resume_with_job(resume_text, job_description, use_api)  # Try again
                
            return 5, "Error in processing match."  # Default score and justification

def find_top_matches(resume_text, job_descriptions, top_n=2):
    """Find the top N matching job descriptions for a resume."""
    matches = []
    
    logger.info("Comparing resume with job descriptions...")
    for job in tqdm(job_descriptions):
        match_score, justification = compare_resume_with_job(resume_text, job, use_api=True)
        matches.append({
            'company': job['company'],
            'role': job['role'],
            'locations': job['locations'],
            'tech_stack': job['tech_stack'],
            'match_percentage': match_score,
            'justification': justification
        })
        time.sleep(1)  # Rate limiting
    
    # Sort matches by match percentage in descending order
    matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    # Return top N matches
    return matches[:top_n]

def main():
    # File paths
    job_desc_path = "./data/SRN_AI_Engineer_Assessment/SRN_JD/job_descriptions.json"
    resume_dir = "./data/SRN_AI_Engineer_Assessment/resume"
    output_file = "./results/Problem1_b_resume_matches.json"

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load job descriptions
    logger.info("Loading job descriptions...")
    job_descriptions = load_job_descriptions(job_desc_path)
    
    if not job_descriptions:
        logger.error("No job descriptions loaded. Exiting.")
        return

    # Process each resume
    results = []
    resume_files = glob.glob(f"{resume_dir}/*.pdf")
    
    logger.info(f"Processing {len(resume_files)} resumes...")
    for resume_path in tqdm(resume_files):
        resume_name = os.path.basename(resume_path)
        logger.info(f"Processing resume: {resume_name}")
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(resume_path)
        if not resume_text:
            logger.warning(f"Could not extract text from {resume_name}")
            continue
        
        # Find top matches
        top_matches = find_top_matches(resume_text, job_descriptions)
        
        # Store results
        results.append({
            'resume': resume_name,
            'matches': top_matches
        })

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
