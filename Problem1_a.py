from dotenv import load_dotenv
import os
import PyPDF2
import csv
import glob
import openai
from openai import OpenAI
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import openpyxl
import logging

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
                extracted_text = page.extract_text()
                if extracted_text:  # Check if text extraction was successful
                    text += extracted_text + "\n"
                else:
                    logger.warning(f"Could not extract text from a page in {pdf_path}")
        
        if not text.strip():
            logger.warning(f"Extracted text is empty for {pdf_path}")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def preprocess_resume_text(text):
    """Clean and preprocess resume text."""
    if not text:
        return ""
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that aren't relevant
    text = re.sub(r'[^\w\s.@,()-]', '', text)
    
    # Remove PDF artifacts often found in extracted text
    text = re.sub(r'(?i)pdf|adobe|acrobat', '', text)
    
    return text.strip()

def extract_job_description_from_url(url):
    """Extract job description from a URL using OpenAI's GPT-4o."""
    try:
        # Add User-Agent to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Send GET request to the URL
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Get the HTML content
        html_content = response.text
        
        # Limit the HTML content to a reasonable size to avoid token limits
        html_content = html_content[:30000]  # Limit to 30K characters
        
        # Use OpenAI's GPT-4o to extract the job description
        prompt = f"""
        You are an AI assistant tasked with extracting job descriptions from webpages.
        
        I'll provide you with the HTML content of a webpage that contains a job posting.
        Your task is to extract the complete job description, including:
        - Job title
        - Company information
        - Job responsibilities
        - Required qualifications
        - Benefits
        - Any other relevant information
        
        Please provide ONLY the extracted text without any additional commentary or explanations.
        Extract as much relevant information as possible.
        
        HTML content:
        {html_content}
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o model
            messages=[
                {"role": "system", "content": "You are an AI assistant that extracts job descriptions from HTML content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
            max_tokens=2000
        )
        
        # Get the extracted job description
        job_desc = response.choices[0].message.content.strip()
        
        if not job_desc:
            logger.warning(f"Could not extract job description from {url}")
            
        return job_desc
    except Exception as e:
        logger.error(f"Error extracting job description from {url}: {e}")
        return ""

def extract_job_descriptions_from_excel(excel_path):
    """
    Extract job URLs from Excel file and fetch job descriptions.
    Saves the extracted descriptions back to the Excel file to avoid re-extraction.
    """
    try:
        # Read the Excel file using pandas
        df = pd.read_excel(excel_path)
        
        # Check if 'Job Description' column exists, if not, add it
        if 'Job Description' not in df.columns:
            df['Job Description'] = None
        
        # Get the URLs from the first column (adjust if needed)
        job_urls = df.iloc[:, 0].tolist()
        
        # Create job descriptions by fetching content from URLs
        job_descriptions = []
        modified = False  # Flag to track if we modified the Excel file
        
        logger.info(f"Processing {len(job_urls)} job URLs...")
        for index, url in enumerate(tqdm(job_urls)):
            # Skip if URL is empty or invalid
            if pd.isna(url) or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                logger.warning(f"Skipping invalid URL at index {index}: {url}")
                continue
            
            # Get additional information from Excel
            title = df.iloc[index, 1] if df.shape[1] > 1 and not pd.isna(df.iloc[index, 1]) else "Unknown Title"
            company = df.iloc[index, 2] if df.shape[1] > 2 and not pd.isna(df.iloc[index, 2]) else "Unknown Company"
            
            # Check if we already have a job description
            existing_description = df.loc[index, 'Job Description']
            job_desc_text = ""
            
            if pd.isna(existing_description) or existing_description == "":
                # No existing description, fetch from URL
                logger.info(f"Extracting job description for {title} at {company} (URL: {url[:50]}...)")
                job_desc_text = extract_job_description_from_url(url)
                
                if not job_desc_text:
                    logger.warning(f"Could not extract job description from {url}, using dummy text")
                    job_desc_text = f"Job at {company}. Please refer to the original job posting for more details."
                
                # Save the extracted description back to the Excel file
                df.loc[index, 'Job Description'] = job_desc_text
                modified = True
            else:
                # Use existing description
                logger.info(f"Using existing job description for {title} at {company}")
                job_desc_text = existing_description
            
            # Creating a structured job description
            job_desc = {
                'id': index,
                'url': url,
                'title': title,
                'organization': company,
                'full_description': job_desc_text
            }
            
            job_descriptions.append(job_desc)
            
            # Add a small delay only when extracting new descriptions
            if modified and index < len(job_urls) - 1:  # Not needed for the last URL
                time.sleep(1)
        
        # Save the modified Excel file if we extracted any new descriptions
        if modified:
            logger.info(f"Saving updated job descriptions back to {excel_path}")
            df.to_excel(excel_path, index=False)
        
        logger.info(f"Successfully processed {len(job_descriptions)} job descriptions")
        return job_descriptions
    except Exception as e:
        logger.error(f"Error processing job descriptions from {excel_path}: {e}")
        return []

def get_embedding(text, model="text-embedding-3-small"):
    """Get embeddings for text using OpenAI's embedding model."""
    try:
        # Clean and truncate text if needed
        text = text.replace("\n", " ")
        
        # Get embedding from OpenAI
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        # Return the embedding
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Implement exponential backoff for rate limits
        if "rate limit" in str(e).lower():
            wait_time = 5
            logger.info(f"Rate limit exceeded. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return get_embedding(text, model)  # Try again
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
            {job_description['full_description'][:3500]}  # Also limiting job description text
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
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
    else:
        # Using embeddings and cosine similarity
        resume_embedding = get_embedding(resume_text[:8000])  # Limiting text for API constraints
        job_embedding = get_embedding(job_description['full_description'][:8000])
        
        if resume_embedding and job_embedding:
            similarity = cosine_similarity(resume_embedding, job_embedding)
            # Convert similarity score to a scale from 1-10 (similarity ranges from -1 to 1)
            score = (similarity + 1) * 5  # Scale to 1-10
            justification = f"Based on semantic similarity analysis with a cosine similarity of {similarity:.2f}."
            return score, justification
        else:
            return 5, "Could not generate embeddings for comparison."  # Default score and justification

def find_top_matches(resume_text, job_descriptions, top_n=2, use_api=True):
    """Find the top N matching jobs for a resume."""
    matches = []
    
    # Preprocess resume text to improve matching
    processed_resume = preprocess_resume_text(resume_text)
    
    logger.info("Comparing resume with job descriptions...")
    for job in tqdm(job_descriptions):
        match_score, justification = compare_resume_with_job(processed_resume, job, use_api=use_api)
        matches.append({
            'job': job,
            'match_percentage': match_score,
            'justification': justification
        })
    
    # Sort matches by match percentage in descending order
    matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    # Return top N matches
    return matches[:top_n]

def save_results_to_csv(results, output_file):
    """Save matching results to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Resume', 'Rank', 'Job Title', 'Organization', 'Match Score', 'Match Justification', 'Job URL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for resume, matches in results.items():
            for i, match in enumerate(matches):
                writer.writerow({
                    'Resume': resume,
                    'Rank': i+1,
                    'Job Title': match['job']['title'],
                    'Organization': match['job']['organization'],
                    'Match Score': f"{match['match_percentage']:.1f}/10",
                    'Match Justification': match['justification'],
                    'Job URL': match['job']['url']
                })
    
    logger.info(f"Results saved to {output_file}")

def main():
    # File paths
    resume_dir = "data/SRN_AI_Engineer_Assessment/resume"
    jobs_excel_path = "data/SRN_AI_Engineer_Assessment/Paraform_Jobs.xlsx"
    output_csv = f"./results/Problem1_a_resume_matches.csv"
    
    # Decide whether to use API or embeddings for comparison
    use_api = True  # Set to False to use embeddings instead
    
    # Check if directories exist
    if not os.path.exists(resume_dir):
        logger.error(f"Resume directory not found: {resume_dir}")
        return
    
    if not os.path.exists(jobs_excel_path):
        logger.error(f"Jobs Excel file not found: {jobs_excel_path}")
        return
    
    # Get all PDF files in the resume directory
    resume_paths = glob.glob(f"{resume_dir}/*.pdf")
    
    if not resume_paths:
        logger.warning(f"No PDF files found in {resume_dir}")
        return
    
    # Extract job descriptions from Excel
    job_descriptions = extract_job_descriptions_from_excel(jobs_excel_path)
    
    if not job_descriptions:
        logger.error("No job descriptions found. Exiting.")
        return
    
    logger.info(f"Found {len(resume_paths)} PDF files and {len(job_descriptions)} job descriptions.")
    
    # Dictionary to store all results
    all_results = {}
    
    # Process each resume
    for resume_path in resume_paths:
        # Get the resume filename
        filename = os.path.basename(resume_path)
        logger.info(f"\nProcessing resume: {filename}")
        
        # Extract text from resume
        resume_text = extract_text_from_pdf(resume_path)
        
        if not resume_text:
            logger.warning(f"Could not extract text from {filename}. Skipping.")
            continue
        
        # Find top matching jobs
        top_matches = find_top_matches(resume_text, job_descriptions, use_api=use_api)
        
        # Store results
        all_results[filename] = top_matches
        
        # Display results
        logger.info(f"\nTop matches for {filename}:")
        for i, match in enumerate(top_matches):
            job = match['job']
            score = match['match_percentage']
            justification = match['justification']
            logger.info(f"{i+1}. {job['title']} at {job['organization']} - {score:.1f}/10 match")
            logger.info(f"   Justification: {justification}")
    
    # Save all results to CSV
    save_results_to_csv(all_results, output_csv)

if __name__ == "__main__":
    main()
