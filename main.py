import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from collections import Counter
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from io import BytesIO
from arxiv_retriever import search_arxiv
from chroma_db import ChromaVectorDB
from llm_generation import LLMLiteratureGenerator
from config import MAX_RESULTS, START_YEAR, END_YEAR, OUTPUT_DIR, RUN_ID, SIMILARITY_THRESHOLD


def safe_json_serializer(obj):
    if hasattr(obj, 'dtype'):
        if pd.api.types.is_integer_dtype(obj):
            return int(obj)
        elif pd.api.types.is_float_dtype(obj):
            return float(obj)
        else:
            return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

def analyze_trends(papers):
    if not papers:
        return None
        
    years = []
    for p in papers:
        try:
            year = int(p.get('published', 0))
            years.append(year)
        except (ValueError, TypeError):
            continue
    
    if not years:
        return None
        
    year_counts = Counter(years)
    sorted_items = sorted(year_counts.items())
    sorted_years = [item[0] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]
    
    min_year = min(years) if years else "N/A"
    max_year = max(years) if years else "N/A"
    
    return {
        'total_papers': len(papers),
        'year_range': f"{min_year}-{max_year}",
        'papers_per_year': dict(year_counts),
        'sorted_years': sorted_years,
        'sorted_counts': sorted_counts,
        'avg_papers_per_year': len(papers) / len(set(years)) if years else 0,
        'latest_year_count': year_counts.get(max_year, 0) if years else 0,
        'min_year': min_year,
        'max_year': max_year,
        'config_year_range': f"{START_YEAR}-{END_YEAR}",
        'all_years_found': sorted(years)  # Add this for debugging
    }

def generate_pdf(review_content, query, papers_count, year_range):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor='#2E4057'
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=12,
            textColor='#2E4057'
        )
        normal_style = styles['Normal']
        
        story = []
        
        title_text = f"Literature Review: {query}"
        story.append(Paragraph(title_text, title_style))

        meta_text = f"""
        <b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Papers analyzed:</b> {papers_count}<br/>
        <b>Year range:</b> {year_range}<br/>
        <b>Query:</b> {query}
        """
        story.append(Paragraph(meta_text, normal_style))
        story.append(Spacer(1, 20))
        
        lines = review_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 12))
                continue
                
            if line.startswith('# '):
                story.append(Paragraph(line[2:], title_style))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], heading_style))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], heading_style))
            else:
                clean_line = line.replace('**', '').replace('*', '').replace('`', '')
                story.append(Paragraph(clean_line, normal_style))
        
        doc.build(story)
        
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        os.unlink(pdf_path)
        
        return pdf_data
        
    except Exception as e:
        logging.error(f"PDF generation failed: {str(e)}")
        try:
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica", 10)
            
            p.drawString(100, 750, f"Literature Review: {query}")
            p.drawString(100, 735, "PDF generation encountered an error. Content below:")
            
            y_position = 700
            for line in review_content.split('\n'):
                if y_position < 50:  
                    p.showPage()
                    p.setFont("Helvetica", 10)
                    y_position = 750
                
                p.drawString(100, y_position, line[:80])  
                y_position -= 15
            
            p.save()
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e2:
            logging.error(f"PDF fallback also failed: {str(e2)}")
            return None

if not os.access(OUTPUT_DIR, os.W_OK):
    st.error(f"❌ Cannot write to output directory: {OUTPUT_DIR}")
    st.info("💡 Create the directory or check permissions:")
    st.code(f"mkdir -p {OUTPUT_DIR} && chmod 755 {OUTPUT_DIR}")
    st.stop()

st.set_page_config(
    page_title="RAG Research Assistent",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 RAG Research Assistent")
st.markdown(f"""
**Research analysis with retrieval, trend detection, and interactive visualization.**  
*Configured for years: {START_YEAR}-{END_YEAR} | Max results: {MAX_RESULTS} | Model: Gemma3:4b*
""")

if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'search_stats' not in st.session_state:
    st.session_state.search_stats = {}

with st.sidebar:
    st.header("🔧 Configuration")
    
    st.info(f"""
    **Current Settings:**
    - Years: {START_YEAR}-{END_YEAR}
    - Max Results: {MAX_RESULTS}
    - Model: Gemma3:4b
    """)
    
    with st.expander("Search Settings"):
        custom_max_results = st.slider("Max Results", 5, 30, MAX_RESULTS)
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 0.9, SIMILARITY_THRESHOLD, 0.1)
        
    if 'db' in st.session_state:
        try:
            if 'papers' in st.session_state:
                st.metric("Papers in Session", len(st.session_state.papers))
        except:
            pass
            
    st.divider()
    st.header("💡 How to Use")
    st.markdown("""
    1. **Enter** your research question
    2. **Click** "Search Papers" 
    3. **Review** automated analysis
    4. **Edit** metrics if needed
    5. **Download** comprehensive report
    """)
    
    st.divider()
    st.subheader("🎯 Example Queries")
    examples = [
        "Transformer architecture improvements",
        "Few-shot learning language models", 
        "Neural network compression techniques",
        "Self-supervised learning computer vision",
        "Reinforcement learning large language models"
    ]
    
    for ex in examples:
        if st.button(f"`{ex}`", key=f"ex_{ex[:10]}"):
            st.session_state.query_input = ex
            st.rerun()
    
    if st.session_state.query_history:
        st.divider()
        st.subheader("📚 Recent Queries")
        for i, q in enumerate(st.session_state.query_history[-5:]):
            cols = st.columns([3, 1])
            cols[0].caption(q[:40] + "..." if len(q) > 40 else q)
            if cols[1].button("↻", key=f"reuse_{i}"):
                st.session_state.query_input = q
                st.rerun()

query = st.text_area(
    f"**Enter your research query (searching {START_YEAR}-{END_YEAR}):**",
    height=100,
    placeholder="e.g., efficient transformers for edge devices, recent advances in few-shot learning...",
    key="query_input",
    help=f"Be specific for better results. Searching papers from {START_YEAR} to {END_YEAR}."
)

if st.button("🔍 Search & Analyze Papers", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a research query.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing database...")
            if 'db' not in st.session_state:
                st.session_state.db = ChromaVectorDB()
            db = st.session_state.db
            progress_bar.progress(10)

            status_text.text(f"Searching arXiv ({START_YEAR}-{END_YEAR}) with topic matching...")
            papers = search_arxiv(query, custom_max_results)
            progress_bar.progress(40)
            
            if not papers:
                st.warning(f"❌ No relevant papers found between {START_YEAR}-{END_YEAR}.")
                st.info(f"""
                💡 Try these improvements:
                - Use more specific technical terms
                - Include methodology names
                - Specify your application domain
                - Current year range: {START_YEAR}-{END_YEAR}
                """)
                st.stop()

            avg_relevance = sum(p.get('match_score', 0) for p in papers) / len(papers)
            high_relevance = len([p for p in papers if p.get('match_score', 0) >= 0.7])

            status_text.text("Processing papers...")
            db.add_papers(papers)
            progress_bar.progress(60)

            status_text.text("Generating comprehensive literature review...")
            generator = LLMLiteratureGenerator()
            review = generator.generate_review(papers, query)
            progress_bar.progress(90)

            papers_data = []
            for paper in papers:
                papers_data.append({
                    "title": paper['title'],
                    "year": paper['published'],
                    "relevance_score": paper.get('match_score', 0),
                    "authors": ', '.join(paper['authors'][:3]) + ('...' if len(paper['authors']) > 3 else '')
                })

            df = pd.DataFrame(papers_data)

            trends = analyze_trends(papers)

            st.session_state.update({
                "papers": papers,
                "review": review,
                "df": df,
                "trends": trends,
                "query": query,
                "search_time": datetime.now().strftime("%H:%M:%S"),
                "relevance_stats": {
                    "avg_relevance": avg_relevance,
                    "high_relevance_count": high_relevance,
                    "total_papers": len(papers)
                }
            })
            st.session_state.query_history.append(query)
            st.session_state.search_stats[query] = {
                "papers_found": len(papers),
                "timestamp": datetime.now().isoformat(),
                "avg_relevance": avg_relevance
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            if trends and trends.get('min_year') and trends.get('max_year'):
                year_range_display = f"{trends['min_year']}-{trends['max_year']}"
                latest_count = trends['latest_year_count']
                latest_year = trends['max_year']
                config_range = f"{START_YEAR}-{END_YEAR}"
                
                range_info = f" (Configured: {config_range})" if config_range != year_range_display else ""
            else:
                year_range_display = "N/A"
                latest_count = 0
                latest_year = "N/A"
                range_info = ""

            st.success(f"""
            **✅ Found {len(papers)} relevant papers!**

            **📊 Relevance Quality:**
            - **Average Relevance Score:** {avg_relevance:.2f}/1.0
            - **High Relevance Papers:** {high_relevance} papers (score ≥ 0.7)
            - **Year Range Found:** {year_range_display}{range_info}
            - **Latest Research:** {latest_count} papers in {latest_year}
            """)
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Analysis failed")
            st.error(f"""
            **❌ Analysis failed:** {str(e)}
            
            **Troubleshooting tips:**
            - Check Ollama server is running: `ollama serve`
            - Verify Gemma3 model: `ollama list`
            - Try a simpler query
            - Verify internet connection
            """)

if "papers" in st.session_state:
    papers = st.session_state.papers
    review = st.session_state.review
    df = st.session_state.df
    trends = st.session_state.trends
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Literature Review", "📊 Analysis", "📄 Papers", "💾 Export"])
    
    with tab1:
        st.subheader("Literature Review")
        if review.startswith("[ERROR]") or review.startswith("# Literature Review Generation Failed"):
            st.error("Review generation failed")
            st.info("The raw papers are still available in the 'Papers' tab for manual review.")
        else:
            st.markdown(review)
    
    with tab2:
        st.info(f"**Search Configuration:** Years {START_YEAR}-{END_YEAR} | Max Results: {MAX_RESULTS}")

        if 'relevance_stats' in st.session_state:
            rel_stats = st.session_state.relevance_stats
            col_rel1, col_rel2, col_rel3 = st.columns(3)
            with col_rel1:
                st.metric("Avg Relevance", f"{rel_stats['avg_relevance']:.2f}/1.0")
            with col_rel2:
                st.metric("High Relevance", rel_stats['high_relevance_count'])
            with col_rel3:
                st.metric("Total Papers", rel_stats['total_papers'])

        st.subheader("📊 Research Analysis")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("📋 Papers Overview")

            papers_data = []
            for paper in papers:
                papers_data.append({
                    "title": paper['title'],
                    "year": paper['published'],
                    "relevance_score": paper.get('match_score', 0),
                    "authors": ', '.join(paper['authors'][:3]) + ('...' if len(paper['authors']) > 3 else '')
                })
            
            df = pd.DataFrame(papers_data)

            display_columns = ["title", "year", "relevance_score", "authors"]
            
            edited_df = st.data_editor(
                df[display_columns],
                column_config={
                    "title": st.column_config.TextColumn("Title", width="large", disabled=True),
                    "year": st.column_config.NumberColumn("Year", disabled=True),
                    "relevance_score": st.column_config.NumberColumn(
                        "Relevance", 
                        help="How well this paper matches your query (0-1)",
                        format="%.2f"
                    ),
                    "authors": st.column_config.TextColumn("Authors", disabled=True)
                },
                num_rows="fixed",
                height=400,
                key="papers_editor"
            )
        
        with col_right:
            st.subheader("📅 Publication Trends")
            if trends and trends.get('sorted_years') and trends.get('sorted_counts'):
                trend_df = pd.DataFrame({
                    'Year': trends['sorted_years'],
                    'Number of Papers': trends['sorted_counts']
                })
                
                fig_trend = px.bar(
                    trend_df,
                    x='Year',
                    y='Number of Papers',
                    title=f"Papers per Year ({trends['min_year']}-{trends['max_year']})",
                    labels={"Year": "Publication Year", "Number of Papers": "Number of Papers"},
                    color='Number of Papers',
                    color_continuous_scale='blues'
                )
                
                fig_trend.update_layout(
                    height=300, 
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                fig_trend.update_xaxes(type='category', tickangle=45)
                fig_trend.update_yaxes(title_standoff=10)
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Papers", trends['total_papers'])
                    st.metric("Year Range", trends['year_range'])
                with col_stat2:
                    st.metric("Average/Year", f"{trends['avg_papers_per_year']:.1f}")
                    st.metric("Latest Year", trends['latest_year_count'])
                
                if len(trends['sorted_years']) > 1:
                    growth = ((trends['sorted_counts'][-1] - trends['sorted_counts'][0]) / 
                             trends['sorted_counts'][0] * 100) if trends['sorted_counts'][0] > 0 else 0
                    st.metric("Growth Trend", f"{growth:+.1f}%", delta=f"{growth:+.1f}%")
                    
            else:
                st.info("No trend data available")

        st.subheader("📈 Relevance Score Distribution")
        
        if 'relevance_score' in edited_df.columns:
            fig_dist = px.histogram(
                edited_df,
                x="relevance_score",
                nbins=10,
                title="Distribution of Relevance Scores",
                labels={"relevance_score": "Relevance Score", "count": "Number of Papers"},
                color_discrete_sequence=['#1f77b4']
            )
            
            fig_dist.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Relevance Score (0-1)",
                yaxis_title="Number of Papers"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)

            avg_relevance = edited_df['relevance_score'].mean()
            high_relevance_count = len(edited_df[edited_df['relevance_score'] >= 0.7])
            
            col_rel1, col_rel2, col_rel3 = st.columns(3)
            with col_rel1:
                st.metric("Average Relevance", f"{avg_relevance:.3f}")
            with col_rel2:
                st.metric("High Relevance (≥0.7)", high_relevance_count)
            with col_rel3:
                st.metric("Score Range", f"{edited_df['relevance_score'].min():.2f}-{edited_df['relevance_score'].max():.2f}")
    
    with tab3:
        st.subheader("Retrieved Papers")
        search_info = st.session_state.search_stats.get(st.session_state.query, {})
        st.caption(f"Found {len(papers)} papers • Search at {search_info.get('timestamp', 'Unknown')} • Years: {START_YEAR}-{END_YEAR}")
        
        for i, paper in enumerate(papers, 1):
            with st.expander(f"{i}. {paper['title']}", expanded=i==1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                    st.markdown(f"**Year:** {paper['published']}")
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    if paper.get('match_score'):
                        st.markdown(f"**Relevance Score:** {paper['match_score']:.2f}/1.0")
                    
                with col2:
                    st.markdown(f"[📄 arXiv Link]({paper['url']})")
                    citation = paper.get("citation", "")
                    if st.button("📋 Copy Citation", key=f"cite_{i}"):
                        st.code(citation, language="bibtex")
                        st.toast("Citation copied to clipboard!", icon="✅")
    
    with tab4:
        st.subheader("Download Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📄 Literature Review (MD)",
                review,
                "literature_review.md",
                "text/markdown",
                use_container_width=True
            )
        
        with col2:
            if st.button("📊 Generate PDF", use_container_width=True, key="generate_pdf"):
                with st.spinner("Generating PDF..."):
                    pdf_data = generate_pdf(
                        review, 
                        st.session_state.query, 
                        len(papers),
                        trends['year_range'] if trends else f"{START_YEAR}-{END_YEAR}"
                    )
                    
                    if pdf_data:
                        st.download_button(
                            "💾 Download PDF",
                            pdf_data,
                            "literature_review.pdf",
                            "application/pdf",
                            use_container_width=True,
                            key="download_pdf"
                        )
                    else:
                        st.error("Failed to generate PDF")
        
        with col3:
            bibtex_content = "\n\n".join(p.get("citation", "") for p in papers)
            st.download_button(
                "📚 BibTeX References", 
                bibtex_content,
                "references.bib",
                "text/plain",
                use_container_width=True
            )
        
        with col4:
            csv_data = df[["title", "year", "relevance_score", "authors"]].to_csv(index=False)
            st.download_button(
                "💾 Papers Data (CSV)",
                csv_data,
                "papers_analysis.csv",
                "text/csv",
                use_container_width=True
            )
        st.download_button(
            "📦 Full Data (JSON)",
            json.dumps({
                "query": st.session_state.query,
                "timestamp": datetime.now().isoformat(),
                "papers_found": len(papers),
                "papers": papers,
                "trends": trends,
                "config": {
                    "start_year": START_YEAR,
                    "end_year": END_YEAR,
                    "max_results": MAX_RESULTS
                }
            }, indent=2, ensure_ascii=False, default=safe_json_serializer),
            "research_analysis.json",
            "application/json",
            use_container_width=True
        )
        
        st.info("💡 All exports include the in depth analysis of collected papers")

st.divider()
st.caption("🔬 RAG Research Assistant  • all rights reserved 2025 • ")