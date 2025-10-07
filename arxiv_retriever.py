import arxiv
import logging
import re
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from retrying import retry
from config import MAX_RESULTS, START_YEAR, END_YEAR, CACHE_DIR

def escape_bibtex(s):
    if not isinstance(s, str):
        return s
    return s.replace('{', '\\{').replace('}', '\\}').replace('&', '\\&')

def clean_abstract(text):
    if not text:
        return ""

    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)  
    text = re.sub(r'\\[{}]', '', text) 
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    text = ' '.join(text.split())
    return text.strip()

def format_arxiv_citation(paper):
    authors = [a.name for a in paper.authors]
    authors_full = escape_bibtex(' and '.join(authors))  
    

    arxiv_id = paper.entry_id.split('/')[-1]
    key = f"arxiv_{arxiv_id}"
    title = escape_bibtex(paper.title)
    
    return f"""@article{{{key},
  title = {{{title}}},
  author = {{{authors_full}}},
  journal = {{arXiv preprint}},
  year = {{{paper.published.year}}},
  note = {{arXiv:{arxiv_id}}}
}}"""

def expand_query(query):
    expanded_queries = []
    expanded_queries.append(query)
    expanded_queries.append(f'ti:"{query}"')
    expanded_queries.append(f'abs:"{query}"')

    words = query.split()
    if len(words) > 1:
        expanded_queries.append(' AND '.join(f'abs:"{word}"' for word in words))
        expanded_queries.append(f'abs:"{query}"')
    
    return expanded_queries

def matches_topic(paper_content, user_query):
    if not paper_content or not user_query:
        return 0.0
    
    content_lower = paper_content.lower()
    query_lower = user_query.lower()
    
    query_terms = re.findall(r'\b\w+\b', query_lower)
    if not query_terms:
        return 0.0
    exact_match_score = 1.0 if query_lower in content_lower else 0.0

    term_scores = []
    for term in query_terms:
        if len(term) <= 2:  
            continue
        term_pattern = r'\b' + re.escape(term) + r'\b'
        matches = len(re.findall(term_pattern, content_lower))
        if matches > 0:
            term_scores.append(min(matches * 0.2, 0.6)) 

    sentences = re.split(r'[.!?]+', content_lower)
    intro_sentences = sentences[:3]  
    intro_text = ' '.join(intro_sentences)
    
    intro_score = 0.0
    for term in query_terms:
        if len(term) > 2 and term in intro_text:
            intro_score += 0.3
    
    total_score = (exact_match_score * 0.4 + 
                  sum(term_scores) * 0.4 + 
                  min(intro_score, 0.6) * 0.2)
    
    return min(total_score, 1.0)


@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
def search_arxiv_single_query(query, max_results, user_query):
    try:
        date_filter = f"submittedDate:[{START_YEAR}01010000 TO {END_YEAR}12312359]"
        full_query = f"({query}) AND {date_filter}"

        logging.info(f"Searching arXiv with query: {full_query}")
        
        search = arxiv.Search(
            query=full_query,
            max_results=max_results * 3, 
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = []
        seen_ids = set()
        
        for paper in search.results():
            try:
                paper_id = paper.entry_id.split('/')[-1]
                if paper_id in seen_ids:
                    continue
                    
                year = paper.published.year
            
                if 1900 <= year <= datetime.now().year + 1:  
                    clean_title = clean_abstract(paper.title)
                    clean_abstract_text = clean_abstract(paper.summary)
                    paper_content = f"{clean_title}. {clean_abstract_text}"
                    
                    match_score = matches_topic(paper_content, user_query)

                    if match_score >= 0.3: 
                        results.append({
                            'title': paper.title,
                            'abstract': clean_abstract_text,
                            'authors': [a.name for a in paper.authors],
                            'url': paper.entry_id,
                            'published': year,
                            'citation': format_arxiv_citation(paper),
                            'arxiv_id': paper_id,
                            'match_score': match_score  
                        })
                        seen_ids.add(paper_id)
                    
                    if len(results) >= max_results * 2: 
                        break
                        
            except Exception as e:
                logging.warning(f"Error processing paper {paper.entry_id}: {str(e)}")
                continue
                
        logging.info(f"Query '{query}' returned {len(results)} relevant papers")
        return results
        
    except Exception as e:
        logging.error(f"arXiv API error for query '{query}': {str(e)}")
        return []

def search_arxiv(query: str, max_results=MAX_RESULTS):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    if max_results <= 0 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50")

    query_hash = hashlib.md5(f"{query}_{max_results}_{START_YEAR}_{END_YEAR}".encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/arxiv_{query_hash}.pkl"
    cache_duration = timedelta(hours=24)  

    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < cache_duration:
            logging.info(f"Using cached results for query: {query}")
            try:
                cached_results = pickle.load(open(cache_file, 'rb'))
                if cached_results:
                    logging.info(f"Loaded {len(cached_results)} papers from cache")
                    return cached_results
            except Exception as e:
                logging.warning(f"Cache loading failed: {str(e)}")
    
    logging.info(f"Performing fresh search for: '{query}' (max_results: {max_results}, years: {START_YEAR}-{END_YEAR})")

    expanded_queries = expand_query(query)
    all_results = []
    seen_ids = set()
    
    for expanded_query in expanded_queries:
        try:
            logging.info(f"Trying expanded query: {expanded_query}")
            results = search_arxiv_single_query(expanded_query, max_results, query)
            
            for paper in results:
                if paper['arxiv_id'] not in seen_ids:
                    all_results.append(paper)
                    seen_ids.add(paper['arxiv_id'])
                    
            logging.info(f"Expanded query '{expanded_query}' found {len(results)} papers, total so far: {len(all_results)}")
            
            if len(all_results) >= max_results * 2:
                logging.info(f"Reached sufficient papers ({len(all_results)}), stopping query expansion")
                break
                
        except Exception as e:
            logging.warning(f"Query '{expanded_query}' failed: {str(e)}")
            continue

    all_results.sort(key=lambda x: (x.get('match_score', 0), x.get('published', 0)), reverse=True)
    final_results = all_results[:max_results]

    if final_results:
        avg_score = sum(p.get('match_score', 0) for p in final_results) / len(final_results)
        years_found = list(set(p.get('published', 0) for p in final_results))
        years_found.sort()
        
        logging.info(f"✅ Retrieved {len(final_results)} papers for query: '{query}'")
        logging.info(f"   📊 Average match score: {avg_score:.2f}/1.0")
        logging.info(f"   📅 Years found: {years_found}")
        logging.info(f"   🔍 Config year range: {START_YEAR}-{END_YEAR}")
    else:
        logging.warning(f"❌ No relevant papers found for query: '{query}'")
        logging.info(f"   💡 Try: broader terms, different keywords, or adjust year range ({START_YEAR}-{END_YEAR})")

    if final_results:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pickle.dump(final_results, open(cache_file, 'wb'))
            logging.info(f"Cached {len(final_results)} papers to {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to cache results: {str(e)}")
    
    return final_results