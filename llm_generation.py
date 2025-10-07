import requests
import json
import re
import logging
from retrying import retry
from config import OLLAMA_URL, LLM_MODEL, LLM_CONTEXT_WINDOW, LLM_MAX_TOKENS, LLM_TEMPERATURE

class LLMLiteratureGenerator:
    def __init__(self, ollama_url=OLLAMA_URL, model_name=LLM_MODEL):
        self.url = ollama_url
        self.model_name = model_name
        self.max_context = LLM_CONTEXT_WINDOW

        try:
            # Extract base URL for health check
            base_url = '/'.join(ollama_url.split('/')[:3])  # Gets http://localhost:11434
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Verify model is available
            models = response.json().get('models', [])
            model_available = any(model['name'].startswith(model_name) for model in models)
            if not model_available:
                logging.warning(f"Model {model_name} may not be available. Found: {[m['name'] for m in models]}")
            else:
                logging.info(f"Ollama server and model {model_name} are ready")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama server check failed: {str(e)}")
            raise RuntimeError(f"Cannot connect to Ollama server at {ollama_url}: {str(e)}")

    def _estimate_tokens(self, text):
        return len(text) // 4  # Approximation: 1 token ≈ 4 characters

    def _truncate_papers_for_context(self, papers, max_tokens):
        if not papers:
            return []
            
        # Sort papers by relevance indicators
        sorted_papers = sorted(papers, key=lambda x: (
            x.get('match_score', 0),  # Most relevant first
            x.get('published', 0)     # Newer papers first
        ), reverse=True)
        
        selected_papers = []
        current_tokens = 0
        
        for paper in sorted_papers:
            paper_text = f"{paper['title']} {paper['abstract']}"
            paper_tokens = self._estimate_tokens(paper_text)
            
            if current_tokens + paper_tokens <= max_tokens:
                selected_papers.append(paper)
                current_tokens += paper_tokens
            else:
                break
                
        if not selected_papers and papers:
            # If even one paper doesn't fit, take just the title of the most relevant paper
            first_paper = papers[0].copy()
            first_paper['abstract'] = ""  # Remove abstract to save space
            selected_papers = [first_paper]
            
        logging.info(f"Selected {len(selected_papers)}/{len(papers)} papers for context ({current_tokens} tokens)")
        return selected_papers

    def _build_enhanced_literature_review_prompt(self, papers, query):
        
        # Group papers by year for trend analysis
        papers_by_year = {}
        for paper in papers:
            year = paper.get('published', 'Unknown')
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper)
        
        # Create structured paper analysis
        paper_analysis = ""
        for i, paper in enumerate(papers, 1):
            match_score = paper.get('match_score', 0)
            paper_analysis += f"""
## PAPER {i}: {paper['title']}

**Publication Year:** {paper.get('published', 'Unknown')}
**Relevance Score:** {match_score:.2f}/1.0
**Key Contributors:** {', '.join(paper['authors'][:5])}{' et al.' if len(paper['authors']) > 5 else ''}

**Core Contribution:** {paper['abstract'][:300]}... [See full abstract in details below]

**Methodological Approach:** [Analyze the technical approach used]
**Key Findings:** [Extract 2-3 main findings]
**Limitations/Gaps:** [Identify any mentioned limitations]

---"""

        prompt = f"""# ACADEMIC LITERATURE REVIEW GENERATION TASK

## CONTEXT & OBJECTIVE
You are an expert academic research assistant with deep domain knowledge. Your task is to synthesize {len(papers)} research papers into a comprehensive, critical literature review addressing the research query: "{query}".

## PAPERS TO ANALYZE (Sorted by Relevance)
{paper_analysis}

## REVIEW STRUCTURE REQUIREMENTS
Create a professional literature review with the following exact sections:

### 1. Executive Summary & Research Landscape
- Briefly summarize the research area and its significance
- State the main research question derived from "{query}"
- Overview of the field's current state and evolution

### 2. Historical Development & Chronological Trends
- Trace the evolution of research from earliest to latest papers
- Identify key milestones and paradigm shifts
- Analyze how methodologies and focus areas have changed over time

### 3. Thematic Analysis & Methodological Approaches
- Group papers by common themes, methodologies, or approaches
- Compare and contrast different technical approaches
- Identify dominant vs. emerging methodologies

### 4. Critical Synthesis of Key Findings
- Synthesize major contributions and discoveries
- Highlight contradictory findings or ongoing debates
- Identify consensus areas and disputed claims

### 5. Gap Analysis & Research Challenges
- Systematically identify underexplored areas
- Analyze methodological limitations across studies
- Highlight practical or theoretical challenges

### 6. Future Research Directions
- Propose specific, actionable research questions
- Suggest methodological improvements
- Identify promising emerging areas

### 7. Conclusion & Overall Assessment
- Summarize the field's maturity and trajectory
- Assess the quality and robustness of existing research
- Provide final recommendations for researchers

## CRITICAL THINKING GUIDELINES
- **Be analytical, not descriptive**: Don't just summarize papers; analyze patterns, relationships, and implications
- **Identify connections**: Show how papers relate to each other (building upon, contradicting, extending)
- **Evaluate evidence quality**: Comment on methodological rigor, sample sizes, validation approaches
- **Maintain academic tone**: Use precise, formal academic language
- **Be comprehensive but concise**: Cover all important aspects without unnecessary repetition

## TECHNICAL REQUIREMENTS
- Use proper academic markdown formatting with headings and bullet points
- Cite papers using their titles and years throughout the text
- Include all papers in the reference section using provided citations
- Maintain objective, evidence-based analysis

## OUTPUT FORMAT
Generate ONLY the literature review content in markdown format, following the exact section structure above. Do not include any introductory text or disclaimers.

BEGIN LITERATURE REVIEW:"""

        return prompt

    def _build_comprehensive_prompt(self, papers, query):
        years = [p.get('published', 0) for p in papers if p.get('published')]
        year_range = f"{min(years)}-{max(years)}" if years else "Unknown"

        paper_list = ""
        for i, p in enumerate(papers):
            match_score = p.get('match_score', 0)
            paper_list += f"""
**{i+1}. {p['title']}** ({p.get('published', 'Unknown')})
- *Authors:* {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}
- *Relevance:* {match_score:.2f}/1.0
- *Key Focus:* {p['abstract'][:200]}...
"""
        
        prompt = f"""# COMPREHENSIVE LITERATURE REVIEW: "{query.upper()}"

## RESEARCH CONTEXT
- **Query Focus:** {query}
- **Timespan Analyzed:** {year_range}
- **Papers Synthesized:** {len(papers)} key publications
- **Domain:** Computer Science / AI Research

## PAPER CORPUS ANALYSIS
The following papers represent the most relevant research in this domain:
{paper_list}

## SYNTHESIS INSTRUCTIONS
Create a master-level literature review that:

1. **Contextualizes** the research area within broader computer science
2. **Analyzes** methodological trends and evolution
3. **Critiques** the strength of evidence and claims
4. **Identifies** theoretical and practical implications
5. **Proposes** concrete future research avenues

## REQUIRED ANALYSIS DEPTH
- **Technical Depth:** Explain methodologies with appropriate technical detail
- **Comparative Analysis:** Use tables or comparative frameworks where helpful
- **Critical Evaluation:** Assess limitations and strengths of each approach
- **Synthesis Quality:** Demonstrate deep understanding of interconnections

## OUTPUT STRUCTURE
# Literature Review: [Innovative Title Capturing Essence]

### Abstract
[150-word summary capturing key insights]

### 1. Introduction and Background
[Context, significance, research questions]

### 2. Methodological Landscape
[Technical approaches, algorithms, frameworks]

### 3. Thematic Analysis
[Major themes, sub-areas, focus areas]

### 4. Critical Evaluation
[Strengths, limitations, validation approaches]

### 5. Research Gaps and Future Directions
[Specific, actionable research opportunities]

### 6. Conclusion
[Synthesis of key insights and field assessment]

### References
[Complete citations for all analyzed papers]

Generate only the review content in clean markdown:"""

        return prompt

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def _make_llm_request(self, prompt, temperature=0.2, max_tokens=4000):
        """Make LLM request with context management."""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": min(self.max_context, LLM_CONTEXT_WINDOW),
                "num_predict": min(max_tokens, LLM_MAX_TOKENS)
            }
        }
        
        try:
            response = requests.post(self.url, json=data, timeout=200)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.Timeout:
            logging.error("LLM request timed out")
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM request failed: {str(e)}")
            raise

    def generate_review(self, papers, query):
        if not papers:
            logging.warning("No papers provided")
            return "No papers provided."

        # Validate papers
        required_keys = {"title", "abstract", "authors", "published", "url"}
        for i, p in enumerate(papers):
            if not all(k in p for k in required_keys):
                missing = required_keys - set(p.keys())
                raise ValueError(f"Paper {i} missing keys: {missing}")

        try:
            context_papers = self._truncate_papers_for_context(papers, self.max_context // 2)
     
            if len(papers) > 8 or len(query.split()) > 4:

                review_prompt = self._build_comprehensive_prompt(context_papers, query)
                temperature = 0.3
                max_tokens = 4000
            else:
                review_prompt = self._build_enhanced_literature_review_prompt(context_papers, query)
                temperature = 0.25
                max_tokens = 3500
            
            logging.info(f"Using prompt strategy for {len(papers)} papers, query: '{query}'")
            
            # Generate literature review
            review = self._make_llm_request(review_prompt, temperature=temperature, max_tokens=max_tokens)
            
            # Post-process the review
            review = self._post_process_review(review, query, papers)
            
            logging.info(f"Successfully generated review ({len(review)} characters)")
            return review

        except Exception as e:
            logging.error(f"LLM generation failed: {str(e)}")
            fallback_review = self._create_fallback_review(papers, query, str(e))
            return fallback_review

    def _post_process_review(self, review, query, papers):
        review = review.strip()

        if not review.startswith('#') and not review.startswith('#'):

            review = f"# Literature Review: {query}\n\n{review}"

        required_sections = [
            'Introduction', 'Background', 'Method', 'Analysis', 
            'Discussion', 'Conclusion', 'Future', 'Reference'
        ]

        if 'Reference' not in review and 'References' not in review:
            references_section = "\n\n## References\n\n" + "\n".join([
                f"{i+1}. {p['title']} ({p.get('published', 'Unknown')}) - {', '.join(p['authors'][:3])}"
                for i, p in enumerate(papers)
            ])
            review += references_section
        
        return review

    def _create_fallback_review(self, papers, query, error_msg):
        papers_by_year = {}
        for paper in papers:
            year = paper.get('published', 'Unknown')
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper)

        paper_items = []
        for p in papers:
            match_score = p.get('match_score', 0)
            paper_items.append(f"- **{p['title']}** ({p.get('published', 'Unknown')}) - Relevance: {match_score:.2f}")
 
        fallback_content = f"""# Literature Review: {query}

*Note: This review was automatically generated from paper analysis due to LLM processing limitations.*

## Executive Summary
This analysis synthesizes {len(papers)} research papers relevant to "{query}". The papers span from {min(papers_by_year.keys()) if papers_by_year else 'Unknown'} to {max(papers_by_year.keys()) if papers_by_year else 'Unknown'}.

## Key Papers Analyzed
{chr(10).join(paper_items)}

## Temporal Distribution
- **Earliest work:** {min(papers_by_year.keys()) if papers_by_year else 'N/A'}
- **Latest work:** {max(papers_by_year.keys()) if papers_by_year else 'N/A'}
- **Total timespan:** {int(max(papers_by_year.keys())) - int(min(papers_by_year.keys())) if papers_by_year and len(papers_by_year) > 1 else 'N/A'} years

## Methodological Trends
Based on abstract analysis, the research approaches include:
- Various machine learning and AI techniques
- Computational methods and algorithms
- Theoretical frameworks and empirical studies

## Research Gaps Identified
- Need for more comprehensive comparative studies
- Opportunities for interdisciplinary approaches
- Potential for practical applications and real-world validation

## Conclusion
The analyzed papers provide substantial foundation for research in "{query}". Future work should focus on integrating findings across studies and addressing identified gaps.

## Error Information
*LLM Generation Issue: {error_msg}*

## References
{chr(10).join([f"{i+1}. {p['citation']}" for i, p in enumerate(papers) if p.get('citation')])}
"""
        
        return fallback_content