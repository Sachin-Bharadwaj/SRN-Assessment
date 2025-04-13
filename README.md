# Problem1_a : Match all 3 resumes (data/SRN_AI_Engineer_Assessment/resume) with top2 job postings in .data/SRN_AI_Engineer_Assessment/Paraform_Jobs.xlsx

- The code is present in file: Problem1_a.py and the corresponding results are in ./results/Problem1_a_resume_matches.csv



# Problem1_b : Match all 3 resumes (data/SRN_AI_Engineer_Assessment/resume) with top2 job postings present at SRN internal board (https://app.synapserecruiternetwork.com/curated_list/1739556025152x298945208153276400)

- For this there is a separate scraper written (browser-use.py) which uses an open source browse-use agent to find all the companies mentioned internal to SRN job board and fetches the job description and saves it to the file (./data/SRN_AI_Engineer_Assessment/SRN_JD/job_descriptions.json). This data is then read by the main script file (Problem1_b.py) which matches resumes with the SRN job postings.
- The code is present in file: Problem1_b.py and the corresponding results are in ./results/Problem1_b_resume_matches.json

# Problem2: Candidate matching + Outreach

- There are three LinkedIn scraper files written but I have tested on linkedInProfileExtractor.py which simulates using the limited data (./data/SRN_AI_Engineer_Assessment/JuiceboxExport_1743820890826.csv) and the LLM. The synthetic data is added back as columns in the file (./data/SRN_AI_Engineer_Assessment/Paraform_Jobs.xlsx) towards the end. Problem2.py then finds the top 5-10 best candidate match given the company name, job description and cancidate resume . The output is present in the following file : (./results/matching_candidates_Probook_AI.csv)