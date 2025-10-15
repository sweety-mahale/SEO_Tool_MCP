import os
import json
import httpx
import re
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from urllib.parse import urlparse
import uvicorn
from groq import Groq
from gdrive_mcp import GoogleDriveSEOExporter

# ===================== CONFIGURATION =====================

load_dotenv()
GROQ_MODEL = "llama-3.3-70b-versatile"

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set GROQ_API_KEY in .env file!")

client = Groq(api_key=api_key)

# ===================== FASTMCP SETUP =====================

mcp = FastMCP("SEO_Analyzer_MCP_Server")


# -------Helper Functions-------

def normalize_url(url: str) -> str:
    """Ensure URL has a scheme (http/https)."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_keywords_with_frequency(text: str) -> Dict[str, int]:
    """Extract keywords and their frequencies from text."""
    stop_words = {
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'was', 'are',
        'been', 'has', 'had', 'were', 'will', 'would', 'could', 'should', 'may',
        'can', 'but', 'not', 'you', 'your', 'they', 'their', 'which', 'who', 'what',
        'when', 'where', 'why', 'how', 'all', 'each', 'some', 'more', 'than', 'into',
        'through', 'about', 'over', 'after', 'before', 'between', 'under', 'also'
    }

    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    word_freq: Dict[str, int] = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


def calculate_keyword_density(text: str, keyword: str) -> float:
    """Calculate keyword density percentage."""
    if not keyword:
        return 0.0

    text_lower = text.lower()
    keyword_lower = keyword.lower()
    keyword_count = text_lower.count(keyword_lower)
    words = text_lower.split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    density = (keyword_count / total_words) * 100
    return round(density, 2)


def calculate_readability_score(text: str) -> Dict:
    """Calculate readability metrics"""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()
    total_words = len(words)
    total_sentences = len(sentences)

    if total_sentences == 0 or total_words == 0:
        return {
            "score": 0,
            "grade_level": "Unknown",
            "avg_sentence_length": 0,
            "avg_word_length": 0,
        }

    def count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiou"
        syllable_count = 0
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        return max(1, syllable_count)

    total_syllables = sum(count_syllables(word) for word in words)
    avg_sentence_length = total_words / total_sentences
    avg_syllables_per_word = total_syllables / total_words

    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    flesch_score = max(0, min(100, flesch_score))

    if flesch_score >= 90:
        grade = "Very Easy (5th grade)"
    elif flesch_score >= 80:
        grade = "Easy (6th grade)"
    elif flesch_score >= 70:
        grade = "Fairly Easy (7th grade)"
    elif flesch_score >= 60:
        grade = "Standard (8th-9th grade)"
    elif flesch_score >= 50:
        grade = "Fairly Difficult (10th-12th grade)"
    elif flesch_score >= 30:
        grade = "Difficult (College)"
    else:
        grade = "Very Difficult (College graduate)"

    avg_word_length = sum(len(word) for word in words) / total_words

    return {
        "score": round(flesch_score, 1),
        "grade_level": grade,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "avg_word_length": round(avg_word_length, 1),
        "total_sentences": total_sentences,
        "total_words": total_words,
    }


def assess_content_quality(text: str, word_count: int) -> Dict:
    """Assess content quality based on various factors."""
    if word_count >= 2000:
        length_score = 100
    elif word_count >= 1000:
        length_score = 80
    elif word_count >= 500:
        length_score = 60
    elif word_count >= 300:
        length_score = 40
    else:
        length_score = 20

    words = text.lower().split()
    unique_words = len(set(words))
    unique_ratio = (unique_words / len(words) * 100) if words else 0

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    quality_score = (
        length_score * 0.4 + min(unique_ratio, 100) * 0.3 + min(paragraph_count * 5, 100) * 0.3
    )

    return {
        "score": round(quality_score, 1),
        "length_score": length_score,
        "unique_word_ratio": round(unique_ratio, 1),
        "paragraph_count": paragraph_count,
        "vocabulary_richness": "High" if unique_ratio > 60 else "Medium" if unique_ratio > 40 else "Low",
    }


async def fetch_competitor_keywords(competitor_url: str) -> List[str]:
    """Fetch and extract keywords from competitor page."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=10.0, headers={"User-Agent": "FastMCP-SEO-Bot/1.0"}
        ) as comp_client:
            resp = await comp_client.get(competitor_url)
            soup = BeautifulSoup(resp.text, "lxml")
            body_text = soup.get_text(separator=" ", strip=True)
            word_freq = extract_keywords_with_frequency(body_text)
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            return [kw[0] for kw in top_keywords]
    except Exception:
        return []


def find_keyword_gap(your_keywords: List[str], competitor_keywords: List[str]) -> Dict:
    """Identify keyword gaps between your content and competitors."""
    your_set = set(your_keywords)
    competitor_set = set(competitor_keywords)

    missing_keywords = list(competitor_set - your_set)[:10]
    common_keywords = list(your_set & competitor_set)[:10]
    unique_keywords = list(your_set - competitor_set)[:10]

    return {
        "missing_keywords": missing_keywords,
        "common_keywords": common_keywords,
        "unique_keywords": unique_keywords,
        "gap_percentage": round((len(missing_keywords) / len(competitor_keywords) * 100) if competitor_keywords else 0, 1),
    }


async def analyze_seo(
    url: str,
    primary_keyword: Optional[str] = None,
    competitor_url: Optional[str] = None,
    business_goals: Optional[Dict] = None,
) -> Dict:
    """Core SEO analysis logic with comprehensive metrics and business goals."""
    try:
        url = normalize_url(url)
        if competitor_url:
            competitor_url = normalize_url(competitor_url)

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=15.0, headers={"User-Agent": "FastMCP-SEO-Bot/1.0"}
        ) as seo_client:
            resp = await seo_client.get(url)
            response_time = resp.elapsed.total_seconds()
            soup = BeautifulSoup(resp.text, "lxml")

    except Exception as e:
        raise RuntimeError(f"Failed to fetch or parse the URL: {e}")

    # Basic SEO Metrics
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    title_length = len(title)

    meta_desc_tag = soup.find("meta", {"name": "description"})
    meta_description = meta_desc_tag["content"].strip() if meta_desc_tag and meta_desc_tag.get("content") else ""
    meta_desc_length = len(meta_description)

    meta_robots = soup.find("meta", {"name": "robots"})
    robots_content = meta_robots["content"] if meta_robots and meta_robots.get("content") else "Not set"

    canonical_tag = soup.find("link", {"rel": "canonical"})
    canonical_url = canonical_tag["href"] if canonical_tag and canonical_tag.get("href") else "Not set"

    # Heading Structure
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h2s = [h.get_text(strip=True) for h in soup.find_all("h2")]
    h3s = [h.get_text(strip=True) for h in soup.find_all("h3")]

    # Content Analysis
    body_text = soup.get_text(separator=" ", strip=True)
    word_count = len(body_text.split())

    word_freq = extract_keywords_with_frequency(body_text)
    content_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    top_keywords_list = [kw[0] for kw in content_keywords]

    # Keyword Density & Relevance
    keyword_density = 0.0
    keyword_in_title = False
    keyword_in_meta = False
    keyword_in_h1 = False
    keyword_relevance_score = 0

    if primary_keyword:
        keyword_density = calculate_keyword_density(body_text, primary_keyword)
        keyword_in_title = primary_keyword.lower() in title.lower()
        keyword_in_meta = primary_keyword.lower() in meta_description.lower()
        keyword_in_h1 = any(primary_keyword.lower() in h1.lower() for h1 in h1s)

        relevance_score = 0
        if keyword_in_title:
            relevance_score += 30
        if keyword_in_meta:
            relevance_score += 20
        if keyword_in_h1:
            relevance_score += 25
        if 1.0 <= keyword_density <= 3.0:
            relevance_score += 25
        elif keyword_density > 0:
            relevance_score += 10

        keyword_relevance_score = relevance_score

    # Readability & Content Quality
    readability_metrics = calculate_readability_score(body_text)
    content_quality_metrics = assess_content_quality(body_text, word_count)

    # Competitor Analysis
    competitor_keywords: List[str] = []
    keyword_gap_analysis = None

    if competitor_url:
        competitor_keywords = await fetch_competitor_keywords(competitor_url)
        keyword_gap_analysis = find_keyword_gap(top_keywords_list, competitor_keywords)

    # Image Optimization
    images = soup.find_all("img")
    total_images = len(images)
    images_without_alt = len([img for img in images if not img.get("alt")])
    images_with_alt = total_images - images_without_alt

    # Link Analysis
    links = soup.find_all("a", href=True)
    internal_links: List[str] = []
    external_links: List[str] = []
    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc

    for link in links:
        href = link.get("href", "")
        if href.startswith("http"):
            link_domain = urlparse(href).netloc
            if base_domain in link_domain:
                internal_links.append(href)
            else:
                external_links.append(href)
        elif href.startswith("/"):
            internal_links.append(href)

    # Schema Markup
    schema_scripts = soup.find_all("script", {"type": "application/ld+json"})
    has_schema = len(schema_scripts) > 0

    # Open Graph & Twitter Cards
    og_title = soup.find("meta", {"property": "og:title"})
    og_description = soup.find("meta", {"property": "og:description"})
    og_image = soup.find("meta", {"property": "og:image"})

    has_open_graph = bool(og_title or og_description or og_image)

    # Mobile Optimization
    viewport_tag = soup.find("meta", {"name": "viewport"})
    has_viewport = bool(viewport_tag)

    # Performance
    page_size_kb = len(resp.content) / 1024

    # Issues & Suggestions
    issues: List[str] = []
    suggestions: List[str] = []

    if not title:
        issues.append("❌ Missing <title> tag")
        suggestions.append("Add a descriptive title tag (50-60 characters)")
    elif title_length < 30:
        issues.append("⚠️ Title too short")
        suggestions.append(f"Expand to 50-60 characters (current: {title_length})")
    elif title_length > 60:
        issues.append("⚠️ Title too long")
        suggestions.append(f"Shorten to 50-60 characters (current: {title_length})")

    if not meta_description:
        issues.append("❌ Missing meta description")
        suggestions.append("Add meta description (140-160 characters)")

    if len(h1s) == 0:
        issues.append("❌ No <h1> tag found")
        suggestions.append("Add H1 tag with primary keyword")
    elif len(h1s) > 1:
        issues.append("⚠️ Multiple <h1> tags found")
        suggestions.append(f"Use only one H1 tag (found {len(h1s)})")

    # Business Goals Section
    business_goals_info = ""
    if business_goals:
        business_goals_info = f"""
🎯 BUSINESS GOALS:
• Objective: {business_goals.get('primary_objective', 'N/A')}
• Audience: {business_goals.get('target_audience', 'N/A')}
• Conversion: {business_goals.get('conversion_goal', 'N/A')}
• Strategy: {business_goals.get('content_strategy', 'N/A')}
• Stage: {business_goals.get('business_stage', 'N/A')}
• Position: {business_goals.get('competitive_position', 'N/A')}
"""

    keyword_gap_info = ""
    if keyword_gap_analysis:
        keyword_gap_info = f"""
🎯 COMPETITOR ANALYSIS:
• Missing Keywords: {keyword_gap_analysis['missing_keywords'][:10]}
• Gap: {keyword_gap_analysis['gap_percentage']}%
"""

    prompt = f"""
You are an expert SEO consultant. Analyze and provide recommendations in JSON format.

URL: {url}
Primary Keyword: {primary_keyword or "Not specified"}

{business_goals_info}

METRICS:
• Title: "{title}" ({title_length} chars)
• Meta: "{meta_description}" ({meta_desc_length} chars)
• Keyword Density: {keyword_density}%
• Word Count: {word_count}
• Readability: {readability_metrics['score']}/100
• Quality: {content_quality_metrics['score']}/100

{keyword_gap_info}

ISSUES: {len(issues)}
{chr(10).join(issues)}

Provide recommendations in this JSON format:
{{
    "seo_score": 75,
    "score_breakdown": {{
        "keyword_optimization": 60,
        "content_quality": 70,
        "readability": 80,
        "technical_seo": 75
    }},
    "score_explanation": "Brief explanation",
    "business_goal_alignment": {{
        "alignment_score": 65,
        "strengths": ["strength1", "strength2", "strength3"],
        "gaps": ["gap1", "gap2"],
        "recommendations": ["rec1", "rec2"]
    }},
    "title_suggestions": ["title1", "title2", "title3"],
    "meta_description_suggestions": ["meta1", "meta2", "meta3"],
    "content_improvements": ["imp1", "imp2", "imp3"],
    "technical_fixes": ["fix1", "fix2", "fix3"],
    "conversion_optimization": ["cro1", "cro2", "cro3"],
    "priority_actions": ["CRITICAL: action1", "HIGH: action2"]
}}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        llm_text = response.choices[0].message.content.strip()
        if llm_text.startswith("```json"):
            llm_text = llm_text.split("```json")[1].split("```")[0].strip()
        elif llm_text.startswith("```"):
            llm_text = llm_text.split("```")[1].split("```")[0].strip()

        llm_output = json.loads(llm_text)
    except Exception:
        llm_output = {"seo_score": 0, "score_explanation": "Error parsing AI response"}

    return {
        "url": url,
        "primary_keyword": primary_keyword,
        "competitor_url": competitor_url,
        "business_goals": business_goals,
        "metrics": {
            "meta_tags": {
                "title": title,
                "title_length": title_length,
                "meta_description": meta_description,
                "meta_description_length": meta_desc_length,
                "canonical_url": canonical_url,
                "robots": robots_content,
            },
            "keyword_analysis": {
                "primary_keyword": primary_keyword,
                "density": keyword_density,
                "in_title": keyword_in_title,
                "in_meta_description": keyword_in_meta,
                "in_h1": keyword_in_h1,
                "relevance_score": keyword_relevance_score,
                "top_keywords": top_keywords_list[:15],
            },
            "content": {
                "word_count": word_count,
                "h1_count": len(h1s),
                "h2_count": len(h2s),
                "h3_count": len(h3s),
                "quality_score": content_quality_metrics["score"],
                "vocabulary_richness": content_quality_metrics["vocabulary_richness"],
            },
            "readability": {
                "score": readability_metrics["score"],
                "grade_level": readability_metrics["grade_level"],
            },
            "competitor_analysis": keyword_gap_analysis,
            "images": {
                "total": total_images,
                "with_alt": images_with_alt,
                "without_alt": images_without_alt,
            },
            "links": {
                "internal": len(internal_links),
                "external": len(external_links),
            },
            "technical": {
                "page_size_kb": round(page_size_kb, 2),
                "response_time_seconds": round(response_time, 2),
                "has_viewport": has_viewport,
                "has_schema": has_schema,
                "has_open_graph": has_open_graph,
            },
        },
        "issues_found": issues,
        "suggestions": suggestions,
        "ai_recommendations": llm_output,
    }

@mcp.tool()
async def seo_analyzer(url: str, primary_keyword: Optional[str] = None, competitor_url: Optional[str] = None, business_goals: Optional[Dict] = None) -> Dict:
    """Analyze SEO metrics and provide recommendations based on business goals."""
    return await analyze_seo(url, primary_keyword, competitor_url, business_goals)

@mcp.tool()
async def export_seo_to_google_drive(
    seo_data: Dict,
    website_name: Optional[str] = None,
    folder_id: Optional[str] = None
) -> Dict:
    """
    Export SEO analysis results to Google Drive as a Word document.
    
    Args:
        seo_data: SEO analysis results from seo_analyzer tool
        website_name: Name for the website (e.g., "amazon", "shopify")
        folder_id: Google Drive folder ID to upload to (optional)
    
    Returns:
        Dictionary with upload status, filename, and Google Drive links
    
    Example:
        # First analyze
        analysis = await seo_analyzer("https://amazon.com")
        
        # Then export to Google Drive
        result = await export_seo_to_google_drive(
            seo_data=analysis,
            website_name="amazon",
            folder_id="1a2b3c4d5e6f7g8h9i0j"
        )
    """
    try:
        exporter = GoogleDriveSEOExporter()
        result = exporter.create_and_upload_seo_report(seo_data, website_name, folder_id)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to export to Google Drive: {str(e)}"
        }
    
@mcp.tool()
async def list_gdrive_folders() -> Dict:
    """
    List all folders in Google Drive to find folder IDs.
    
    Returns:
        List of folders with their IDs and names
    
    Example:
        folders = await list_gdrive_folders()
        # Returns: {'success': True, 'folders': [{'id': '...', 'name': 'SEO Reports'}], 'count': 1}
    """
    try:
        exporter = GoogleDriveSEOExporter()
        folders = exporter.list_folders()
        return {
            'success': True,
            'folders': folders,
            'count': len(folders)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    
@mcp.tool()
async def create_gdrive_folder(folder_name: str, parent_folder_id: Optional[str] = None) -> Dict:
    """
    Create a new folder in Google Drive for organizing SEO reports.
    
    Args:
        folder_name: Name for the new folder (e.g., "SEO Reports 2025")
        parent_folder_id: ID of parent folder (optional)
    
    Returns:
        Dictionary with folder_id and success status
    
    Example:
        result = await create_gdrive_folder("SEO Reports 2025")
        # Use the returned folder_id when exporting reports
    """
    try:
        exporter = GoogleDriveSEOExporter()
        folder_id = exporter.create_folder(folder_name, parent_folder_id)
        return {
            'success': True,
            'folder_id': folder_id,
            'folder_name': folder_name,
            'message': f"Folder '{folder_name}' created successfully"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    

# ===================== FASTAPI SETUP =====================
app = FastAPI(title="SEO Analyzer")

class AnalyzeRequest(BaseModel):
    url: str
    primary_keyword: Optional[str] = None
    competitor_url: Optional[str] = None
    business_goals: Dict

class ExportRequest(BaseModel):
    website_name: Optional[str] = None
    folder_id: Optional[str] = None

@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict:
    try:
        result = await analyze_seo(req.url, req.primary_keyword, req.competitor_url, req.business_goals)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SEO analysis failed: {str(e)}")

@app.post("/export-to-drive")
async def export_to_drive(req: Request) -> Dict:
   """Export the last SEO analysis to Google Drive"""
   try:
        data = await req.json()
        seo_data = data.get('seo_data')
        website_name = data.get('website_name')
        folder_id = data.get('folder_id')
        
        if not seo_data:
            raise HTTPException(status_code=400, detail="seo_data is required")
        
        exporter = GoogleDriveSEOExporter()
        result = exporter.create_and_upload_seo_report(seo_data, website_name, folder_id)
        return result
   except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")


@app.get("/gdrive-folders")
async def get_folders():
    """List all Google Drive folders"""
    try:
        exporter = GoogleDriveSEOExporter()
        folders = exporter.list_folders()
        return {"success": True, "folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>SEO Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        h1 { color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 1.1em; }
        
        /* Form Section Styles */
        .form-section { 
            background: #f8f9fa; 
            padding: 25px; 
            border-radius: 8px; 
            border-left: 4px solid #667eea; 
            margin-bottom: 30px; 
        }
        .section-title { 
            font-size: 1.3em; 
            color: #333; 
            margin-bottom: 20px; 
            font-weight: 600; 
        }
        
        .form-group { margin-bottom: 20px; }
        label { display: block; font-weight: 600; margin-bottom: 8px; color: #333; font-size: 14px; }
        input[type="text"], input[type="url"] { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 14px; transition: border-color 0.3s; }
        input:focus { outline: none; border-color: #667eea; }
        
        .question-group { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .question-label { 
            font-weight: 600; 
            color: #333; 
            margin-bottom: 12px; 
            display: block; 
            font-size: 1em; 
        }
        .radio-group { display: flex; flex-direction: column; gap: 10px; }
        .radio-option { display: flex; align-items: center; padding: 12px; background: #f9f9f9; border: 2px solid #e0e0e0; border-radius: 6px; cursor: pointer; transition: all 0.3s; }
        .radio-option:hover { border-color: #667eea; background: #f5f7ff; }
        .radio-option input[type="radio"] { margin-right: 12px; cursor: pointer; width: 18px; height: 18px; accent-color: #667eea; }
        .radio-option label { cursor: pointer; flex: 1; font-size: 0.95em; color: #555; }
        .radio-option.selected { border-color: #667eea; background: #f5f7ff; }
        .radio-option input[type="radio"]:checked + label { color: #667eea; font-weight: 600; }
        
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 14px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; width: 100%; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        button:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .btn-export { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .btn-export:hover { box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4); }
        
        .loading { text-align: center; padding: 40px; color: #666; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        #result { margin-top: 30px; }
        .section { background: #f9f9f9; padding: 20px; margin-bottom: 20px; border-radius: 8px; border-left: 4px solid #667eea; }
        .section h2 { color: #333; margin-bottom: 15px; font-size: 1.3em; }
        .metric { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e0e0e0; }
        .metric:last-child { border-bottom: none; }
        .metric-label { font-weight: 600; color: #555; }
        .metric-value { color: #333; }
        
        .score-box { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }
        .score-number { font-size: 3.5em; font-weight: bold; margin: 10px 0; }
        .score-breakdown { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .score-item { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .score-item-label { color: #666; font-size: 0.9em; margin-bottom: 5px; }
        .score-item-value { font-size: 2em; font-weight: bold; color: #667eea; }
        
        .issue { background: #fff3cd; padding: 12px 15px; margin: 8px 0; border-radius: 6px; border-left: 3px solid #ffc107; }
        .suggestion { background: #d4edda; padding: 12px 15px; margin: 8px 0; border-radius: 6px; border-left: 3px solid #28a745; }
        .keyword-badge { display: inline-block; background: #667eea; color: white; padding: 6px 12px; margin: 4px; border-radius: 6px; font-size: 0.9em; }
        .error-box { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 8px; color: #721c24; }
        
        .priority-critical { border-left-color: #dc3545; background: #f8d7da; }
        .priority-high { border-left-color: #ffc107; background: #fff3cd; }
        .priority-medium { border-left-color: #17a2b8; background: #d1ecf1; }
        
        .business-goals-section { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .success-box { background: #d4edda; border: 1px solid #c3e6cb; padding: 20px; border-radius: 8px; color: #155724; margin: 20px 0; }
        .error-box { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 8px; color: #721c24; margin: 20px 0; }
        
        .gdrive-link { display: inline-block; background: #4285f4; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; margin: 10px 5px; transition: background 0.3s; }
        .gdrive-link:hover { background: #357ae8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 AI SEO Analyzer</h1>
        <p class="subtitle">Comprehensive SEO analysis with AI recommendations aligned to your business goals</p>
        
        <form id="seoForm">
            <div class="form-section">
                <h2 class="section-title">📊 Website & Business Information</h2>
                
                <div class="form-group">
                    <label>Website URL *</label>
                    <input type="text" id="url" placeholder="https://example.com" required>
                </div>
                
                <div class="form-group">
                    <label>Primary Keyword (Optional)</label>
                    <input type="text" id="keyword" placeholder="e.g., SEO tools, digital marketing">
                </div>
                
                <div class="form-group">
                    <label>Competitor URL (Optional)</label>
                    <input type="text" id="competitor" placeholder="https://competitor.com">
                </div>

                <hr style="margin: 30px 0; border: none; border-top: 2px solid #e0e0e0;">

                <h3 style="color: #667eea; margin: 25px 0 15px 0; font-size: 1.2em;">🎯 Business Goals</h3>
                <p style="color: #666; margin-bottom: 20px; font-size: 0.95em;">Help us tailor SEO recommendations to your specific business objectives</p>
                
                <!-- Question 1: Primary Objective -->
                <div class="question-group">
                    <span class="question-label">1. What is your primary business objective?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="primary_objective" value="increase_sales" id="obj1">
                            <label for="obj1">Increase product/service sales</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="primary_objective" value="lead_generation" id="obj2">
                            <label for="obj2">Generate qualified leads</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="primary_objective" value="brand_awareness" id="obj3">
                            <label for="obj3">Build brand awareness and authority</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="primary_objective" value="traffic_growth" id="obj4">
                            <label for="obj4">Grow organic traffic</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="primary_objective" value="engagement" id="obj5">
                            <label for="obj5">Increase user engagement and retention</label>
                        </div>
                    </div>
                </div>

                <!-- Question 2: Target Audience -->
                <div class="question-group">
                    <span class="question-label">2. Who is your target audience?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="target_audience" value="b2b_enterprise" id="aud1">
                            <label for="aud1">B2B - Enterprise/Corporate decision-makers</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="target_audience" value="b2b_smb" id="aud2">
                            <label for="aud2">B2B - Small to medium businesses</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="target_audience" value="b2c_general" id="aud3">
                            <label for="aud3">B2C - General consumers</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="target_audience" value="b2c_niche" id="aud4">
                            <label for="aud4">B2C - Specific niche/hobbyist community</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="target_audience" value="mixed" id="aud5">
                            <label for="aud5">Mixed audience (B2B + B2C)</label>
                        </div>
                    </div>
                </div>

                <!-- Question 3: Conversion Goal -->
                <div class="question-group">
                    <span class="question-label">3. What is your main conversion goal?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="conversion_goal" value="purchase" id="conv1">
                            <label for="conv1">Direct purchase/transaction</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="conversion_goal" value="form_submission" id="conv2">
                            <label for="conv2">Form submission/Contact request</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="conversion_goal" value="signup" id="conv3">
                            <label for="conv3">Newsletter signup/Account creation</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="conversion_goal" value="consultation" id="conv4">
                            <label for="conv4">Book consultation/Demo request</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="conversion_goal" value="download" id="conv5">
                            <label for="conv5">Content download (ebook, whitepaper, etc.)</label>
                        </div>
                    </div>
                </div>

                <!-- Question 4: Content Strategy -->
                <div class="question-group">
                    <span class="question-label">4. What is your content strategy focus?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="content_strategy" value="educational" id="cont1">
                            <label for="cont1">Educational content (tutorials, guides, how-tos)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="content_strategy" value="product_focused" id="cont2">
                            <label for="cont2">Product-focused (features, benefits, comparisons)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="content_strategy" value="thought_leadership" id="cont3">
                            <label for="cont3">Thought leadership (insights, trends, analysis)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="content_strategy" value="entertainment" id="cont4">
                            <label for="cont4">Entertainment/Lifestyle content</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="content_strategy" value="news_updates" id="cont5">
                            <label for="cont5">News and industry updates</label>
                        </div>
                    </div>
                </div>

                <!-- Question 5: Business Stage -->
                <div class="question-group">
                    <span class="question-label">5. What stage is your business in?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="business_stage" value="startup" id="stage1">
                            <label for="stage1">Startup (0-2 years, establishing presence)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="business_stage" value="growth" id="stage2">
                            <label for="stage2">Growth (2-5 years, scaling operations)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="business_stage" value="established" id="stage3">
                            <label for="stage3">Established (5+ years, optimizing & expanding)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="business_stage" value="enterprise" id="stage4">
                            <label for="stage4">Enterprise (large-scale, multiple products/markets)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="business_stage" value="pivot" id="stage5">
                            <label for="stage5">Pivoting/Rebranding</label>
                        </div>
                    </div>
                </div>

                <!-- Question 6: Competitive Position -->
                <div class="question-group">
                    <span class="question-label">6. How do you view your competitive position?</span>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="competitive_position" value="market_leader" id="comp1">
                            <label for="comp1">Market leader (maintaining dominance)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="competitive_position" value="challenger" id="comp2">
                            <label for="comp2">Challenger (competing with leaders)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="competitive_position" value="niche_player" id="comp3">
                            <label for="comp3">Niche player (specialized focus)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="competitive_position" value="new_entrant" id="comp4">
                            <label for="comp4">New entrant (building market presence)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="competitive_position" value="disruptor" id="comp5">
                            <label for="comp5">Disruptor (innovative approach)</label>
                        </div>
                    </div>
                </div>
            </div>
            
            <button type="submit" id="submitBtn">🚀 Analyze SEO</button>
        </form>
        
        <div id="exportSection" style="display: none; margin-top: 20px;">
            <div class="form-group">
                <label>Website Name (Optional)</label>
                <input type="text" id="website_name" placeholder="e.g., amazon, shopify">
            </div>
            <button id="exportBtn" class="btn-export">📤 Export to Google Drive</button>
        </div>
        
        <div id="result"></div>
    </div>
        

    <script>
        let currentAnalysisData = null;
                        
        // Radio button styling
        document.querySelectorAll('.radio-option').forEach(option => {
            option.addEventListener('click', function() {
                const radio = this.querySelector('input[type="radio"]');
                radio.checked = true;
                
                const name = radio.name;
                document.querySelectorAll(`input[name="${name}"]`).forEach(r => {
                    r.closest('.radio-option').classList.remove('selected');
                });
                this.classList.add('selected');
            });
        });

        document.getElementById('seoForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Collect business goals with correct key names
            const businessGoals = {
                primary_objective: document.querySelector('input[name="primary_objective"]:checked')?.value,
                target_audience: document.querySelector('input[name="target_audience"]:checked')?.value,
                conversion_goal: document.querySelector('input[name="conversion_goal"]:checked')?.value,
                content_strategy: document.querySelector('input[name="content_strategy"]:checked')?.value,
                business_stage: document.querySelector('input[name="business_stage"]:checked')?.value,
                competitive_position: document.querySelector('input[name="competitive_position"]:checked')?.value
            };
            
            // Validate business goals
            if (!businessGoals.primary_objective || !businessGoals.target_audience || 
                !businessGoals.conversion_goal || !businessGoals.content_strategy || 
                !businessGoals.business_stage || !businessGoals.competitive_position) {
                alert('⚠️ Please complete all Business Goals questions before analyzing!');
                return;
            }
            
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.textContent = '⏳ Analyzing...';
            
            document.getElementById('result').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p style="font-size: 1.1em;">Analyzing SEO metrics....</p>
                    <p style="margin-top: 10px;">This may take 20-40 seconds</p>
                </div>
            `;
            document.getElementById('exportSection').style.display = 'none';
                        
            const url = document.getElementById('url').value;
            const keyword = document.getElementById('keyword').value;
            const competitor = document.getElementById('competitor').value;

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        url: url, 
                        primary_keyword: keyword || null,
                        competitor_url: competitor || null,
                        business_goals: businessGoals
                    })
                });

                const data = await res.json();
                
                if (data.error) {
                    document.getElementById('result').innerHTML = `
                        <div class="error-box">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                } else {
                    currentAnalysisData = data;
                    displayResults(data);
                    document.getElementById('exportSection').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <div class="error-box">
                        <h3>❌ Error</h3>
                        <p>Failed to analyze: ${error.message}</p>
                    </div>
                `;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Analyze SEO';
            }
        });
                        
document.getElementById('exportBtn').addEventListener('click', async function() {
            if (!currentAnalysisData) {
                alert('No analysis data to export!');
                return;
            }

            const exportBtn = document.getElementById('exportBtn');
            exportBtn.disabled = true;
            exportBtn.textContent = '⏳ Exporting to Google Drive...';

            const websiteName = document.getElementById('website_name').value || null;

            try {
                const res = await fetch('/export-to-drive', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        seo_data: currentAnalysisData,
                        website_name: websiteName
                    })
                });

                const result = await res.json();

                if (result.success) {
                    const exportResult = document.createElement('div');
                    exportResult.className = 'success-box';
                    exportResult.innerHTML = `
                        <h3>✅ Export Successful!</h3>
                        <p><strong>Filename:</strong> ${result.filename}</p>
                        <p>${result.message}</p>
                        <div style="margin-top: 15px;">
                            <a href="${result.gdrive_view_link}" target="_blank" class="gdrive-link">
                                📄 View in Google Drive
                            </a>
                            ${result.gdrive_download_link ? 
                                `<a href="${result.gdrive_download_link}" target="_blank" class="gdrive-link">
                                    ⬇️ Download Document
                                </a>` : ''}
                        </div>
                    `;
                    
                    document.getElementById('result').insertBefore(exportResult, document.getElementById('result').firstChild);
                } else {
                    const errorResult = document.createElement('div');
                    errorResult.className = 'error-box';
                    errorResult.innerHTML = `
                        <h3>❌ Export Failed</h3>
                        <p>${result.message || result.error}</p>
                        <p><small>Make sure you have set up Google Drive credentials (credentials.json)</small></p>
                    `;
                    document.getElementById('result').insertBefore(errorResult, document.getElementById('result').firstChild);
                }
            } catch (error) {
                const errorResult = document.createElement('div');
                errorResult.className = 'error-box';
                errorResult.innerHTML = `
                    <h3>❌ Export Error</h3>
                    <p>${error.message}</p>
                `;
                document.getElementById('result').insertBefore(errorResult, document.getElementById('result').firstChild);
            } finally {
                exportBtn.disabled = false;
                exportBtn.textContent = '📤 Export to Google Drive';
            }
        });
                        
        function displayResults(data) {
            const ai = data.ai_recommendations || {};
            const score = ai.seo_score || 0;
            const breakdown = ai.score_breakdown || {};
            const metrics = data.metrics || {};
            const businessGoalAlignment = ai.business_goal_alignment || {};
            
            let html = `
                <div class="score-box">
                    <h2>Overall SEO Score</h2>
                    <div class="score-number">${score}/100</div>
                    <p style="font-size: 1.1em;">${ai.score_explanation || ''}</p>
                </div>

                <div class="score-breakdown">
                    <div class="score-item">
                        <div class="score-item-label">Keyword Optimization</div>
                        <div class="score-item-value">${breakdown.keyword_optimization || 0}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-label">Content Quality</div>
                        <div class="score-item-value">${breakdown.content_quality || 0}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-label">Readability</div>
                        <div class="score-item-value">${breakdown.readability || 0}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-item-label">Technical SEO</div>
                        <div class="score-item-value">${breakdown.technical_seo || 0}</div>
                    </div>
                </div>

                ${businessGoalAlignment.alignment_score ? `
                <div class="business-goals-section">
                    <h2>🎯 Business Goal Alignment</h2>
                    <div style="font-size: 2.5em; font-weight: bold; margin: 15px 0;">${businessGoalAlignment.alignment_score}/100</div>
                    
                    ${businessGoalAlignment.strengths && businessGoalAlignment.strengths.length > 0 ? `
                    <div style="margin-top: 20px;">
                        <h3 style="margin-bottom: 10px;">✅ Strengths:</h3>
                        ${businessGoalAlignment.strengths.map(s => `<div style="background: rgba(255,255,255,0.2); padding: 10px; margin: 5px 0; border-radius: 6px;">• ${s}</div>`).join('')}
                    </div>
                    ` : ''}
                    
                    ${businessGoalAlignment.gaps && businessGoalAlignment.gaps.length > 0 ? `
                    <div style="margin-top: 20px;">
                        <h3 style="margin-bottom: 10px;">⚠️ Gaps to Address:</h3>
                        ${businessGoalAlignment.gaps.map(g => `<div style="background: rgba(255,255,255,0.2); padding: 10px; margin: 5px 0; border-radius: 6px;">• ${g}</div>`).join('')}
                    </div>
                    ` : ''}
                    
                    ${businessGoalAlignment.recommendations && businessGoalAlignment.recommendations.length > 0 ? `
                    <div style="margin-top: 20px;">
                        <h3 style="margin-bottom: 10px;">💡 Strategic Recommendations:</h3>
                        ${businessGoalAlignment.recommendations.map(r => `<div style="background: rgba(255,255,255,0.2); padding: 10px; margin: 5px 0; border-radius: 6px;">• ${r}</div>`).join('')}
                    </div>
                    ` : ''}
                </div>
                ` : ''}

                <div class="section">
                    <h2>🎯 Priority Actions</h2>
                    ${(ai.priority_actions || []).map(action => {
                        const priority = action.includes('CRITICAL') ? 'priority-critical' : 
                                       action.includes('HIGH') ? 'priority-high' : 'priority-medium';
                        return `<div class="suggestion ${priority}">${action}</div>`;
                    }).join('')}
                </div>

                ${ai.conversion_optimization && ai.conversion_optimization.length > 0 ? `
                <div class="section">
                    <h2>💰 Conversion Optimization</h2>
                    ${ai.conversion_optimization.map((opt, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${opt}</div>`
                    ).join('')}
                </div>
                ` : ''}

                <div class="section">
                    <h2>⚠️ Issues Found (${data.issues_found?.length || 0})</h2>
                    ${(data.issues_found || []).map(issue => `<div class="issue">${issue}</div>`).join('')}
                </div>

                <div class="section">
                    <h2>✅ Suggested Fixes (${data.suggestions?.length || 0})</h2>
                    ${(data.suggestions || []).map(sugg => `<div class="suggestion">${sugg}</div>`).join('')}
                </div>

                <div class="section">
                    <h2>📝 Meta Tags</h2>
                    <div class="metric">
                        <span class="metric-label">Title</span>
                        <span class="metric-value">${metrics.meta_tags?.title_length || 0} characters</span>
                    </div>
                    <div style="padding: 12px; background: white; margin: 12px 0; border-radius: 6px; border: 1px solid #e0e0e0;">
                        "${metrics.meta_tags?.title || 'No title'}"
                    </div>
                    <div class="metric">
                        <span class="metric-label">Meta Description</span>
                        <span class="metric-value">${metrics.meta_tags?.meta_description_length || 0} characters</span>
                    </div>
                    <div style="padding: 12px; background: white; margin: 12px 0; border-radius: 6px; border: 1px solid #e0e0e0;">
                        "${metrics.meta_tags?.meta_description || 'No description'}"
                    </div>
                </div>

                <div class="section">
                    <h2>🔑 Keyword Analysis</h2>
                    <div class="metric">
                        <span class="metric-label">Primary Keyword</span>
                        <span class="metric-value">${metrics.keyword_analysis?.primary_keyword || 'Not specified'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Keyword Density</span>
                        <span class="metric-value">${metrics.keyword_analysis?.density || 0}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Keyword Relevance Score</span>
                        <span class="metric-value">${metrics.keyword_analysis?.relevance_score || 0}/100</span>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>Top Keywords Found:</strong><br>
                        ${(metrics.keyword_analysis?.top_keywords || []).map(kw => 
                            `<span class="keyword-badge">${kw}</span>`
                        ).join('')}
                    </div>
                </div>

                ${ai.keyword_optimization && ai.keyword_optimization.recommendations ? `
                <div class="section">
                    <h2>💡 Keyword Optimization Tips</h2>
                    ${ai.keyword_optimization.recommendations.map((rec, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${rec}</div>`
                    ).join('')}
                </div>
                ` : ''}

                <div class="section">
                    <h2>📄 Content Metrics</h2>
                    <div class="metric">
                        <span class="metric-label">Word Count</span>
                        <span class="metric-value">${metrics.content?.word_count || 0} words</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Content Quality Score</span>
                        <span class="metric-value">${metrics.content?.quality_score || 0}/100</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Heading Structure</span>
                        <span class="metric-value">H1: ${metrics.content?.h1_count || 0} | H2: ${metrics.content?.h2_count || 0} | H3: ${metrics.content?.h3_count || 0}</span>
                    </div>
                </div>

                ${ai.content_quality_improvements && ai.content_quality_improvements.length > 0 ? `
                <div class="section">
                    <h2>💡 Content Quality Improvements</h2>
                    ${ai.content_quality_improvements.map((imp, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${imp}</div>`
                    ).join('')}
                </div>
                ` : ''}

                <div class="section">
                    <h2>📖 Readability Analysis</h2>
                    <div class="metric">
                        <span class="metric-label">Readability Score</span>
                        <span class="metric-value">${metrics.readability?.score || 0}/100</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Grade Level</span>
                        <span class="metric-value">${metrics.readability?.grade_level || 'N/A'}</span>
                    </div>
                </div>

                ${ai.readability_improvements && ai.readability_improvements.length > 0 ? `
                <div class="section">
                    <h2>💡 Readability Improvements</h2>
                    ${ai.readability_improvements.map((imp, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${imp}</div>`
                    ).join('')}
                </div>
                ` : ''}

                ${metrics.competitor_analysis ? `
                <div class="section">
                    <h2>🏆 Competitor Keyword Gap</h2>
                    <div class="metric">
                        <span class="metric-label">Gap Percentage</span>
                        <span class="metric-value">${metrics.competitor_analysis.gap_percentage}%</span>
                    </div>
                    ${metrics.competitor_analysis.missing_keywords && metrics.competitor_analysis.missing_keywords.length > 0 ? `
                    <div style="margin-top: 15px;">
                        <strong style="color: #dc3545;">⚠️ Missing Keywords (Target These):</strong><br>
                        ${metrics.competitor_analysis.missing_keywords.map(kw => 
                            `<span class="keyword-badge" style="background: #dc3545;">${kw}</span>`
                        ).join('')}
                    </div>
                    ` : ''}
                    ${metrics.competitor_analysis.unique_keywords && metrics.competitor_analysis.unique_keywords.length > 0 ? `
                    <div style="margin-top: 15px;">
                        <strong style="color: #28a745;">✅ Your Unique Keywords:</strong><br>
                        ${metrics.competitor_analysis.unique_keywords.map(kw => 
                            `<span class="keyword-badge" style="background: #28a745;">${kw}</span>`
                        ).join('')}
                    </div>
                    ` : ''}
                </div>
                ` : ''}

                ${ai.competitor_insights && ai.competitor_insights.length > 0 ? `
                <div class="section">
                    <h2>💡 Competitor Insights</h2>
                    ${ai.competitor_insights.map((ins, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${ins}</div>`
                    ).join('')}
                </div>
                ` : ''}

                <div class="section">
                    <h2>🖼️ Image Optimization</h2>
                    <div class="metric">
                        <span class="metric-label">Total Images</span>
                        <span class="metric-value">${metrics.images?.total || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Images with Alt Text</span>
                        <span class="metric-value">${metrics.images?.with_alt || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Images without Alt Text</span>
                        <span class="metric-value">${metrics.images?.without_alt || 0}</span>
                    </div>
                </div>

                <div class="section">
                    <h2>🔗 Link Analysis</h2>
                    <div class="metric">
                        <span class="metric-label">Internal Links</span>
                        <span class="metric-value">${metrics.links?.internal || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">External Links</span>
                        <span class="metric-value">${metrics.links?.external || 0}</span>
                    </div>
                </div>

                <div class="section">
                    <h2>⚙️ Technical SEO</h2>
                    <div class="metric">
                        <span class="metric-label">Page Size</span>
                        <span class="metric-value">${metrics.technical?.page_size_kb || 0} KB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Response Time</span>
                        <span class="metric-value">${metrics.technical?.response_time_seconds || 0}s</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Mobile Viewport</span>
                        <span class="metric-value">${metrics.technical?.has_viewport ? '✅ Yes' : '❌ No'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Schema Markup</span>
                        <span class="metric-value">${metrics.technical?.has_schema ? '✅ Yes' : '❌ No'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Open Graph Tags</span>
                        <span class="metric-value">${metrics.technical?.has_open_graph ? '✅ Yes' : '❌ No'}</span>
                    </div>
                </div>

                ${ai.technical_fixes && ai.technical_fixes.length > 0 ? `
                <div class="section">
                    <h2>💡 Technical Fixes</h2>
                    ${ai.technical_fixes.map((fix, i) => 
                        `<div class="suggestion"><strong>${i + 1}.</strong> ${fix}</div>`
                    ).join('')}
                </div>
                ` : ''}

                ${ai.title_suggestions && ai.title_suggestions.length > 0 ? `
                <div class="section">
                    <h2>💡 Title Tag Suggestions</h2>
                    ${ai.title_suggestions.map((title, i) => 
                        `<div class="suggestion"><strong>Suggestion ${i + 1}:</strong> ${title}</div>`
                    ).join('')}
                </div>
                ` : ''}

                ${ai.meta_description_suggestions && ai.meta_description_suggestions.length > 0 ? `
                <div class="section">
                    <h2>💡 Meta Description Suggestions</h2>
                    ${ai.meta_description_suggestions.map((desc, i) => 
                        `<div class="suggestion"><strong>Suggestion ${i + 1}:</strong> ${desc}</div>`
                    ).join('')}
                </div>
                ` : ''}
            `;
            
            document.getElementById('result').innerHTML = html;
        }
    </script>
</body>
</html>"""
 )

if __name__ == "__main__":
    uvicorn.run("mcp_server_web:app", reload=True)
