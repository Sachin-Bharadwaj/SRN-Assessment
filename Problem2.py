import pandas as pd
import random
import os
import openai
import re
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_job_descriptions(file_path):
    """
    Load job descriptions from Excel file
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        dict: Dictionary mapping company names to job descriptions
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Create a dictionary mapping company names to job descriptions
        job_descriptions = {}
        for _, row in df.iterrows():
            company_name = row.get('Company', '')
            if company_name:
                # Combine all relevant columns into a job description
                description = {
                    'Company': company_name,
                    'Role': row.get('Role', ''),
                    'Job Description': row.get('Job Description', ''),
                    'Tech Stack': row.get('Tech Stack', ''),
                    'Requirements': row.get('Requirements', ''),
                    'YOE': row.get('YOE', ''),
                    'Locations': row.get('Locations', ''),
                    'Industry': row.get('Industry', '')
                }
                
                # Format the job description
                formatted_description = ""
                for key, value in description.items():
                    if pd.notna(value) and value != '':
                        formatted_description += f"{key}: {value}\n\n"
                
                job_descriptions[company_name] = formatted_description
        
        return job_descriptions
    except Exception as e:
        print(f"Error loading job descriptions: {e}")
        return {}

def load_candidate_profiles(file_path):
    """
    Load candidate profiles from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing candidate profiles
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading candidate profiles: {e}")
        return pd.DataFrame()

def select_job_description(job_descriptions):
    """
    Select a job description based on user input or randomly
    
    Args:
        job_descriptions (dict): Dictionary mapping company names to job descriptions
        
    Returns:
        tuple: (company_name, job_description)
    """
    # Get list of company names
    company_names = list(job_descriptions.keys())
    
    # If no companies found
    if not company_names:
        print("No company data found.")
        return None, None
    
    # Print available companies
    print("\nAvailable companies:")
    for idx, name in enumerate(company_names, 1):
        print(f"{idx}. {name}")
    
    # Ask user for company name
    user_input = input("\nEnter the company name (leave empty for random selection): ").strip()
    
    if user_input:
        # Try to find the company name (case-insensitive)
        for company_name in company_names:
            if user_input.lower() == company_name.lower():
                return company_name, job_descriptions[company_name]
        
        # If not found
        print(f"Company '{user_input}' not found. Selecting a random company.")
        company_name = random.choice(company_names)
    else:
        # Random selection
        company_name = random.choice(company_names)
        print(f"Randomly selected company: {company_name}")
    
    return company_name, job_descriptions[company_name]

def prepare_candidate_data(candidates_df):
    """
    Prepare candidate data for GPT processing
    
    Args:
        candidates_df (pandas.DataFrame): DataFrame containing candidate profiles
        
    Returns:
        str: Formatted candidate data,
        dict: Dictionary mapping candidate index to their details
    """
    candidate_data = []
    candidate_details = {}
    
    for idx, row in candidates_df.iterrows():
        # Extract key fields based on actual CSV structure
        try:
            full_name = f"{row.get('First name', '')} {row.get('Last name', '')}"
            linkedin_url = row.get('LinkedIn', '')
            
            candidate_info = {
                "id": idx,
                "name": full_name.strip(),
                "linkedin_url": linkedin_url,
                "title": row.get('Title_GPT', '') or row.get('Current Title', ''),
                "tech_stack": row.get('TechStack_GPT', ''),
                "startup_fit": row.get('StartupFit_GPT', ''),
                "tenure": row.get('Tenure_GPT', ''),
                "current_org": row.get('Current Org Name', ''),
                "education": row.get('Education', '')
            }
            
            # Store candidate details for later use
            candidate_details[idx+1] = {
                "name": full_name.strip(),
                "linkedin_url": linkedin_url
            }
            
            # Format candidate information
            candidate_text = f"Candidate {idx+1}:\n"
            for key, value in candidate_info.items():
                if key not in ['id', 'linkedin_url'] and value and not pd.isna(value):
                    candidate_text += f"- {key.replace('_', ' ').title()}: {value}\n"
                elif key == 'linkedin_url' and value and not pd.isna(value):
                    candidate_text += f"- LinkedIn: {value}\n"
            
            candidate_data.append(candidate_text)
        except Exception as e:
            print(f"Error processing candidate {idx}: {e}")
    
    return "\n".join(candidate_data), candidate_details

def find_matching_candidates(job_description, candidates_data):
    """
    Use GPT-4o to find matching candidates for a job description
    
    Args:
        job_description (str): Job description
        candidates_data (str): Formatted candidate data
        
    Returns:
        str: GPT-4o response with matching candidates and justifications
    """
    try:
        prompt = f"""
        You are an expert AI recruiting assistant. Your task is to find the best 5-10 matching candidate profiles 
        for the following job description. Evaluate candidates based on their title, tech stack, startup fit, and tenure.

        JOB DESCRIPTION:
        {job_description}

        CANDIDATE PROFILES:
        {candidates_data}

        For each candidate, provide:
        1. A match score from 1-10 (where 10 is a perfect match)
        2. A brief justification explaining the match

        Format your response like this:
        1. [Candidate Name](LinkedIn URL) - Score: X/10 - Brief justification for the match (focus on title, tech stack, startup fit, and tenure)
        2. [Candidate Name](LinkedIn URL) - Score: X/10 - Brief justification for the match
        ...

        If a LinkedIn URL is not available, use format:
        1. Candidate Name - Score: X/10 - Brief justification

        Be concise in your justifications but highlight specific matching qualifications.
        List candidates in descending order by match score (best matches first).
        Always include the candidate number at the beginning of each line (e.g., 1., 2., etc.).
        Always include the LinkedIn URL if available in the format [Name](URL).
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI recruiting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT-4o API: {e}")
        return "Failed to process request with GPT-4o. Error: " + str(e)

def parse_gpt_response(response, candidate_details):
    """
    Parse the GPT-4o response to extract candidate information
    
    Args:
        response (str): GPT-4o response
        candidate_details (dict): Dictionary mapping candidate index to their details
        
    Returns:
        list: List of dictionaries containing candidate name, score, LinkedIn URL, and justification
    """
    results = []
    
    # Split the response into lines
    lines = response.strip().split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        try:
            # Extract the candidate number from the enumeration at the beginning
            candidate_num_match = re.match(r'^\s*(\d+)\.\s+', line)
            if not candidate_num_match:
                continue
            
            candidate_num = int(candidate_num_match.group(1))
            
            # Extract name and LinkedIn URL using regex for markdown format [Name](URL)
            name_url_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
            
            if name_url_match:
                name = name_url_match.group(1).strip()
                linkedin_url = name_url_match.group(2).strip()
            else:
                # If no URL format, try to extract just the name
                name_match = re.search(r'^\s*\d+\.\s+([^-]+)-\s+Score', line)
                name = name_match.group(1).strip() if name_match else "Unknown Candidate"
                
                # Try to find the candidate in our details by name
                linkedin_url = ""
                for candidate_id, details in candidate_details.items():
                    if details["name"].lower() == name.lower():
                        linkedin_url = details["linkedin_url"]
                        break
            
            # Extract the score using regex
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)/10', line, re.IGNORECASE)
            if not score_match:
                continue
            
            score = float(score_match.group(1))
            
            # Extract the justification using regex
            justification_match = re.search(r'Score:\s*\d+(?:\.\d+)?/10\s*-\s*(.*)', line, re.IGNORECASE)
            justification = justification_match.group(1).strip() if justification_match else ""
            
            # Create the result dictionary
            candidate_result = {
                "name": name,
                "score": score,
                "linkedin_url": linkedin_url,
                "justification": justification
            }
            
            results.append(candidate_result)
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error details: {e}")
    
    # Sort results by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

def format_results_for_display(results):
    """
    Format the parsed results for console display
    
    Args:
        results (list): List of dictionaries containing candidate information
        
    Returns:
        str: Formatted string for display
    """
    formatted_output = []
    
    for i, result in enumerate(results, 1):
        name = result["name"]
        score = result["score"]
        linkedin_url = result["linkedin_url"]
        justification = result["justification"]
        
        # Format with LinkedIn URL if available
        if linkedin_url:
            formatted_output.append(f"{i}. [{name}]({linkedin_url}) - Score: {score}/10 - {justification}")
        else:
            formatted_output.append(f"{i}. {name} - Score: {score}/10 - {justification}")
    
    return "\n".join(formatted_output)

def generate_outreach_message(candidate_info, job_description, company_name):
    """
    Generate a personalized outreach message for a candidate using GPT-4o
    
    Args:
        candidate_info (dict): Dictionary containing candidate information
        job_description (str): Job description
        company_name (str): Name of the company
        
    Returns:
        str: Personalized outreach message
    """
    try:
        # Extract candidate name
        candidate_name = candidate_info["name"]
        
        # Extract role from job description if possible
        role_match = re.search(r'Role:\s*([^\n]+)', job_description)
        role = role_match.group(1).strip() if role_match else "the open position"
        
        prompt = f"""
        Create a personalized outreach message for a job candidate. The message should be:
        1. Not more than 250 words
        2. Professional but conversational in tone
        3. Mentioning specific qualifications that match the job description
        4. Clear about next steps
        
        COMPANY: {company_name}
        ROLE: {role}
        
        JOB DESCRIPTION SUMMARY:
        {job_description[:500]}...
        
        CANDIDATE INFORMATION:
        Name: {candidate_name}
        Match Score: {candidate_info["score"]}/10
        Match Justification: {candidate_info["justification"]}
        
        The message should:
        - Be addressed to the candidate by name
        - Briefly introduce the company and the role
        - Reference their specific qualifications that make them a good fit based on the justification
        - Have a clear call to action (schedule a call, reply with interest, etc.)
        - Be enthusiastic but not overly familiar
        
        Do not include a subject line - just the body of the message.
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert recruiter crafting personalized outreach messages."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating outreach message: {e}")
        return "Failed to generate outreach message."

def save_to_csv(results, company_name):
    """
    Save the matching candidates to a CSV file
    
    Args:
        results (list): List of dictionaries containing candidate information
        company_name (str): Name of the company
    
    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Create a directory for results if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/matching_candidates_{company_name.replace(' ', '_')}_{timestamp}.csv"
        
        # Write results to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'score', 'linkedin_url', 'justification', 'outreach_message']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        return filename
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return None

def main():
    print("Job Description to Candidate Matcher")
    print("====================================")
    
    # File paths
    jobs_file = "./data/SRN_AI_Engineer_Assessment/Paraform_Jobs.xlsx"
    candidates_file = "./data/SRN_AI_Engineer_Assessment/JuiceboxExport_1743820890826.csv"
    
    # Load job descriptions
    print("Loading job descriptions...")
    job_descriptions = load_job_descriptions(jobs_file)
    
    if not job_descriptions:
        print("No job descriptions found. Exiting.")
        return
    
    # Load candidate profiles
    print("Loading candidate profiles...")
    candidates_df = load_candidate_profiles(candidates_file)
    
    if candidates_df.empty:
        print("No candidate profiles found. Exiting.")
        return
    
    # Select a job description
    company_name, job_description = select_job_description(job_descriptions)
    
    if not job_description:
        print("No job description selected. Exiting.")
        return
    
    print(f"\nSelected Job Description for {company_name}:")
    print("-------------------------------------------")
    print(job_description[:500] + "..." if len(job_description) > 500 else job_description)
    print("-------------------------------------------")
    
    # Prepare candidate data
    print("\nPreparing candidate profiles...")
    candidates_data, candidate_details = prepare_candidate_data(candidates_df)
    
    # Find matching candidates
    print("\nFinding matching candidates using GPT-4o (this may take a moment)...")
    matches_response = find_matching_candidates(job_description, candidates_data)
    
    # Parse the GPT-4o response
    print("\nParsing GPT-4o response...")
    results = parse_gpt_response(matches_response, candidate_details)
    
    # Generate outreach messages for top candidates only (max 10)
    print("\nGenerating personalized outreach messages for top candidates (this may take a moment)...")
    top_candidates = results[:min(10, len(results))]
    for result in top_candidates:
        result['outreach_message'] = generate_outreach_message(result, job_description, company_name)
    
    # For candidates beyond the top 10, set a placeholder message
    for result in results[min(10, len(results)):]:
        result['outreach_message'] = "Outreach message not generated to optimize performance. Generate for top candidates only."
    
    # Format results for display
    formatted_results = format_results_for_display(results)
    
    # Save results to CSV
    csv_file = save_to_csv(results, company_name)
    
    # Display results
    print("\nTop Matching Candidates (with GPT-4o Scores):")
    print("==========================================")
    print(formatted_results)
    
    # Display sample outreach message for the top candidate
    if results:
        print("\nSample Outreach Message for Top Candidate:")
        print("==========================================")
        print(results[0]['outreach_message'])
    
    if csv_file:
        print(f"\nResults and outreach messages saved to: {csv_file}")
        print(f"Outreach messages were generated for the top {len(top_candidates)} candidates.")

if __name__ == "__main__":
    main()
