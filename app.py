import streamlit as st
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from datetime import datetime
import tempfile
import numpy as np

# Try to import optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    st.warning("⚠️ Seaborn not installed. Some advanced visualizations will be unavailable.")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    st.warning("⚠️ ReportLab not installed. PDF generation will use alternative method.")

# Alternative PDF generation using matplotlib
try:
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_PDF_BACKEND = True
except ImportError:
    HAS_PDF_BACKEND = False

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Key Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Gemini API Key not found. Please set it in your .env file as GEMINI_API_KEY.")
    st.stop()

# --- Enhanced Data Processing Function ---
def parse_student_data(json_data):
    """
    Enhanced parsing with additional metrics for better LLM context.
    """
    try:
        submission = json_data[0]
        
        # Overall Test Performance with additional metrics
        overall_performance = {
            "totalTime": submission["test"]["totalTime"],
            "totalQuestions": submission["test"]["totalQuestions"],
            "totalMarks": submission["test"]["totalMarks"],
            "studentTotalTimeTaken": submission["totalTimeTaken"],
            "studentTotalMarkScored": submission["totalMarkScored"],
            "studentTotalAttempted": submission["totalAttempted"],
            "studentTotalCorrect": submission["totalCorrect"],
            "studentOverallAccuracy": submission["accuracy"],
            "timeUtilization": (submission["totalTimeTaken"] / submission["test"]["totalTime"]) * 100,
            "attemptRate": (submission["totalAttempted"] / submission["test"]["totalQuestions"]) * 100,
            "marksPercentage": (submission["totalMarkScored"] / submission["test"]["totalMarks"]) * 100
        }

        # Enhanced Subject-wise Performance
        subject_performance = []
        subject_names_map = {
            "607018ee404ae53194e73d92": "Physics",
            "607018ee404ae53194e73d90": "Chemistry", 
            "607018ee404ae53194e73d91": "Mathematics"
        }
        
        for subject in submission["subjects"]:
            subject_data = {
                "subjectName": subject_names_map.get(subject["subjectId"]["$oid"], "Unknown Subject"),
                "totalTimeTaken": subject["totalTimeTaken"],
                "totalMarkScored": subject["totalMarkScored"],
                "totalAttempted": subject["totalAttempted"],
                "totalCorrect": subject["totalCorrect"],
                "accuracy": subject["accuracy"],
                "avgTimePerQuestion": subject["totalTimeTaken"] / max(subject["totalAttempted"], 1),
                "efficiency": (subject["accuracy"] / max(subject["totalTimeTaken"], 1)) * 100 if subject["totalTimeTaken"] > 0 else 0
            }
            subject_performance.append(subject_data)

        # Enhanced Chapter and Concept Analysis
        chapter_data = {}
        concept_data = {}
        difficulty_analysis = {"easy": {"total": 0, "correct": 0}, "medium": {"total": 0, "correct": 0}, "hard": {"total": 0, "correct": 0}}
        time_accuracy_data = []

        for section in submission["sections"]:
            for question in section["questions"]:
                q_id = question["questionId"]
                chapters = [c["title"] for c in q_id.get("chapters", [])]
                concepts = [c["title"] for c in q_id.get("concepts", [])]
                level = q_id.get("level", "medium").lower()
                time_taken = question.get("timeTaken", 0)
                status = question.get("status", "N/A")

                # Enhanced correctness detection
                is_correct = False
                if question.get("inputValue") and question["inputValue"].get("isCorrect") is not None:
                    is_correct = question["inputValue"]["isCorrect"]
                elif question.get("markedOptions"):
                    for option in question["markedOptions"]:
                        if option.get("isCorrect") is True:
                            is_correct = True
                            break

                # Time vs Accuracy tracking
                time_accuracy_data.append({
                    "time": time_taken,
                    "correct": is_correct,
                    "level": level,
                    "subject": chapters[0] if chapters else "Unknown"
                })

                # Difficulty analysis
                if level in difficulty_analysis:
                    difficulty_analysis[level]["total"] += 1
                    if is_correct:
                        difficulty_analysis[level]["correct"] += 1

                # Enhanced Chapter aggregation
                for chapter in chapters:
                    if chapter not in chapter_data:
                        chapter_data[chapter] = {
                            "totalQuestions": 0, "correctQuestions": 0, "totalTimeTaken": 0,
                            "levels": {}, "avgTimePerQuestion": 0, "strongConcepts": [], "weakConcepts": []
                        }
                    chapter_data[chapter]["totalQuestions"] += 1
                    chapter_data[chapter]["totalTimeTaken"] += time_taken
                    if is_correct:
                        chapter_data[chapter]["correctQuestions"] += 1

                    if level not in chapter_data[chapter]["levels"]:
                        chapter_data[chapter]["levels"][level] = {"total": 0, "correct": 0}
                    chapter_data[chapter]["levels"][level]["total"] += 1
                    if is_correct:
                        chapter_data[chapter]["levels"][level]["correct"] += 1

                # Enhanced Concept aggregation
                for concept in concepts:
                    if concept not in concept_data:
                        concept_data[concept] = {
                            "totalQuestions": 0, "correctQuestions": 0, "totalTimeTaken": 0,
                            "levels": {}, "masteryLevel": "developing"
                        }
                    concept_data[concept]["totalQuestions"] += 1
                    concept_data[concept]["totalTimeTaken"] += time_taken
                    if is_correct:
                        concept_data[concept]["correctQuestions"] += 1

                    if level not in concept_data[concept]["levels"]:
                        concept_data[concept]["levels"][level] = {"total": 0, "correct": 0}
                    concept_data[concept]["levels"][level]["total"] += 1
                    if is_correct:
                        concept_data[concept]["levels"][level]["correct"] += 1

        # Calculate enhanced metrics
        for chapter, data in chapter_data.items():
            data["accuracy"] = (data["correctQuestions"] / data["totalQuestions"]) * 100 if data["totalQuestions"] > 0 else 0
            data["avgTimePerQuestion"] = data["totalTimeTaken"] / data["totalQuestions"] if data["totalQuestions"] > 0 else 0
            
            for level, level_data in data["levels"].items():
                level_data["accuracy"] = (level_data["correct"] / level_data["total"]) * 100 if level_data["total"] > 0 else 0

        for concept, data in concept_data.items():
            data["accuracy"] = (data["correctQuestions"] / data["totalQuestions"]) * 100 if data["totalQuestions"] > 0 else 0
            data["avgTimePerQuestion"] = data["totalTimeTaken"] / data["totalQuestions"] if data["totalQuestions"] > 0 else 0
            
            # Determine mastery level
            if data["accuracy"] >= 80:
                data["masteryLevel"] = "mastered"
            elif data["accuracy"] >= 60:
                data["masteryLevel"] = "developing"
            else:
                data["masteryLevel"] = "needs_attention"
                
            for level, level_data in data["levels"].items():
                level_data["accuracy"] = (level_data["correct"] / level_data["total"]) * 100 if level_data["total"] > 0 else 0

        return {
            "overall_performance": overall_performance,
            "subject_performance": subject_performance,
            "chapter_performance": chapter_data,
            "concept_performance": concept_data,
            "difficulty_analysis": difficulty_analysis,
            "time_accuracy_data": time_accuracy_data
        }

    except Exception as e:
        st.error(f"Error processing JSON data: {e}")
        return None

# --- Enhanced Visualization Functions ---
def create_comprehensive_visualizations(student_data):
    """
    Creates multiple visualizations and returns them as base64 encoded images for PDF.
    """
    plt.style.use('seaborn-v0_8')
    charts = {}
    
    # 1. Subject Performance Radar Chart
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
    subjects = [s["subjectName"] for s in student_data["subject_performance"]]
    accuracies = [s["accuracy"] for s in student_data["subject_performance"]]
    
    angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False)
    accuracies += accuracies[:1]  # Complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, accuracies, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, accuracies, alpha=0.25, color='#2E86AB')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subjects)
    ax.set_ylim(0, 100)
    ax.set_title('Subject Performance Overview', size=16, fontweight='bold')
    ax.grid(True)
    
    charts['subject_radar'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Time vs Accuracy Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    time_acc_data = student_data["time_accuracy_data"]
    
    correct_times = [d["time"] for d in time_acc_data if d["correct"]]
    incorrect_times = [d["time"] for d in time_acc_data if not d["correct"]]
    
    ax.scatter(correct_times, [1]*len(correct_times), alpha=0.6, c='green', label='Correct', s=50)
    ax.scatter(incorrect_times, [0]*len(incorrect_times), alpha=0.6, c='red', label='Incorrect', s=50)
    
    ax.set_xlabel('Time Taken (seconds)', fontsize=12)
    ax.set_ylabel('Correctness', fontsize=12)
    ax.set_title('Time vs Accuracy Analysis', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    charts['time_accuracy'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 3. Chapter Performance Heatmap (with fallback if seaborn not available)
    fig, ax = plt.subplots(figsize=(12, 8))
    chapter_data = student_data["chapter_performance"]
    
    # Create data for heatmap
    chapters = list(chapter_data.keys())
    levels = ['easy', 'medium', 'hard']
    heatmap_data = []
    
    for chapter in chapters:
        row = []
        for level in levels:
            if level in chapter_data[chapter]["levels"]:
                accuracy = chapter_data[chapter]["levels"][level]["accuracy"]
            else:
                accuracy = 0
            row.append(accuracy)
        heatmap_data.append(row)
    
    if HAS_SEABORN:
        sns.heatmap(heatmap_data, 
                    xticklabels=levels, 
                    yticklabels=chapters,
                    annot=True, 
                    fmt='.1f',
                    cmap='RdYlGn',
                    ax=ax,
                    cbar_kws={'label': 'Accuracy (%)'})
    else:
        # Fallback heatmap using matplotlib
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(levels)))
        ax.set_yticks(range(len(chapters)))
        ax.set_xticklabels(levels)
        ax.set_yticklabels(chapters)
        
        # Add text annotations
        for i in range(len(chapters)):
            for j in range(len(levels)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    ax.set_title('Chapter-wise Performance by Difficulty', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    charts['chapter_heatmap'] = fig_to_base64(fig)
    plt.close(fig)
    
    return charts

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.read()).decode()

# --- Enhanced Prompt Engineering ---
def construct_advanced_feedback_prompt(student_data):
    """
    Generates a human-friendly prompt to create deeply personalized academic feedback.
    """
    overall = student_data["overall_performance"]
    subjects = student_data["subject_performance"]
    chapters = student_data["chapter_performance"]
    concepts = student_data["concept_performance"]

    strongest_subject = max(subjects, key=lambda x: x["accuracy"])
    weakest_subject = min(subjects, key=lambda x: x["accuracy"])

    strongest_chapters = sorted(chapters.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:3]
    weakest_chapters = sorted(chapters.items(), key=lambda x: x[1]["accuracy"])[:3]

    time_efficiency = "efficient" if overall["timeUtilization"] < 80 else "needs improvement"

    prompt = f"""
You're a seasoned academic coach and student mentor with over 15 years of experience. Your job is to write detailed, personalized feedback for students, based on their actual performance in tests. This feedback should not only be accurate but also motivating, constructive, and tailored to the student's strengths and areas that need improvement.

Here's the student's performance summary:
- Total Score: {overall['studentTotalMarkScored']} out of {overall['totalMarks']} ({overall['marksPercentage']:.1f}%)
- Accuracy: {overall['studentOverallAccuracy']:.1f}%
- Attempt Rate: {overall['studentTotalAttempted']} out of {overall['totalQuestions']} questions ({overall['attemptRate']:.1f}%)
- Time Used: {overall['timeUtilization']:.1f}% of the allocated time
- Strongest Subject: {strongest_subject['subjectName']} ({strongest_subject['accuracy']:.1f}% accuracy)
- Weakest Subject: {weakest_subject['subjectName']} ({weakest_subject['accuracy']:.1f}% accuracy)

Subject-wise breakdown:
"""
    for subject in subjects:
        prompt += f"""
{subject['subjectName']}:
  - Accuracy: {subject['accuracy']:.1f}%
  - Attempted: {subject['totalAttempted']} questions
  - Avg. Time per Question: {subject['avgTimePerQuestion']:.1f} sec
  - Efficiency Score: {subject['efficiency']:.1f}
"""

    prompt += "\nTop Performing Chapters:\n"
    for chapter, data in strongest_chapters:
        prompt += f"- {chapter}: {data['accuracy']:.1f}% accuracy over {data['totalQuestions']} questions\n"

    prompt += "\nChapters That Need More Focus:\n"
    for chapter, data in weakest_chapters:
        prompt += f"- {chapter}: {data['accuracy']:.1f}% accuracy over {data['totalQuestions']} questions\n"

    prompt += f"""
Time Management Summary:
- Overall Time Usage: {time_efficiency}
- Decision Making Speed: {"Good" if overall['attemptRate'] > 90 else "Needs improvement"}

Now, using this data, write a **personalized feedback report** for the student. The tone should be friendly, conversational, and motivating — as if you're talking directly to the student.

Structure the feedback into six parts:

1. **Personalized Introduction** (150-200 words)
   - Speak to the student directly
   - Acknowledge both strengths and improvement areas
   - Show that you truly understand their performance

2. **Highlight Their Strengths** (100-150 words)
   - Back it up with numbers
   - Explain what this tells us about how they learn best

3. **Focus Areas for Improvement** (200-250 words)
   - Pick 2–3 specific issues (e.g., subjects or chapters)
   - Offer insight on how addressing these can boost their performance

4. **Time Management Tips** (150-200 words)
   - Analyze how they used their time
   - Suggest strategies to improve efficiency without rushing

5. **Action Plan** (200-250 words)
   - Give 3–4 specific, measurable tips they can follow
   - Include weekly and monthly goals

6. **Encouraging Wrap-Up** (100-150 words)
   - End on a positive, hopeful note
   - Reinforce their potential and next steps

Important Notes:
- Don’t be vague — use real performance numbers
- Avoid educational jargon
- Make suggestions feel practical, not overwhelming
- Sound like someone who truly cares about their progress

Write the full feedback report now:
"""

    return prompt

# --- PDF Generation Functions ---
def generate_pdf_report_reportlab(feedback_text, student_data, charts):
    """
    Generate PDF using ReportLab (professional method).
    """
    if not HAS_REPORTLAB:
        return None
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Custom styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2E86AB')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#2E86AB')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Build PDF content
    story = []
    
    # Title
    story.append(Paragraph("Student Performance Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Performance Summary Table
    overall = student_data["overall_performance"]
    summary_data = [
        ['Metric', 'Value'],
        ['Overall Score', f"{overall['studentTotalMarkScored']}/{overall['totalMarks']} ({overall['marksPercentage']:.1f}%)"],
        ['Accuracy', f"{overall['studentOverallAccuracy']:.1f}%"],
        ['Questions Attempted', f"{overall['studentTotalAttempted']}/{overall['totalQuestions']}"],
        ['Time Utilization', f"{overall['timeUtilization']:.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Add charts
    if 'subject_radar' in charts:
        story.append(Paragraph("Performance Visualization", heading_style))
        img_data = base64.b64decode(charts['subject_radar'])
        img = Image(io.BytesIO(img_data), width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
    # Process and add feedback text
    sections = feedback_text.split('**')
    current_text = ""
    
    for i, section in enumerate(sections):
        if section.strip():
            if i % 2 == 1:  # This is a heading
                if current_text.strip():
                    story.append(Paragraph(current_text.strip(), body_style))
                    current_text = ""
                story.append(Paragraph(section.strip(), heading_style))
            else:  # This is body text
                current_text += section
    
    if current_text.strip():
        story.append(Paragraph(current_text.strip(), body_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_pdf_report_matplotlib(feedback_text, student_data, charts):
    """
    Alternative PDF generation using matplotlib (fallback method).
    """
    if not HAS_PDF_BACKEND:
        return None
        
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Title and Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Student Performance Analysis Report', 
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.92, f'Generated on: {datetime.now().strftime("%B %d, %Y")}', 
                fontsize=12, ha='center', transform=ax.transAxes)
        
        # Summary data
        overall = student_data["overall_performance"]
        summary_text = f"""
PERFORMANCE SUMMARY

Overall Score: {overall['studentTotalMarkScored']}/{overall['totalMarks']} ({overall['marksPercentage']:.1f}%)
Accuracy: {overall['studentOverallAccuracy']:.1f}%
Questions Attempted: {overall['studentTotalAttempted']}/{overall['totalQuestions']}
Time Utilization: {overall['timeUtilization']:.1f}%

SUBJECT PERFORMANCE:
"""
        
        for subject in student_data["subject_performance"]:
            summary_text += f"• {subject['subjectName']}: {subject['accuracy']:.1f}% accuracy\n"
        
        ax.text(0.1, 0.8, summary_text, fontsize=11, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Feedback Text
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Process feedback text to fit on page
        lines = feedback_text.replace('**', '').split('\n')
        y_pos = 0.95
        
        for line in lines[:50]:  # Limit to first 50 lines to fit on page
            if line.strip():
                ax.text(0.05, y_pos, line.strip()[:100], fontsize=10, 
                       transform=ax.transAxes, wrap=True)
                y_pos -= 0.03
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Charts
        if charts:
            for chart_name, chart_data in charts.items():
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Decode and display chart
                img_data = base64.b64decode(chart_data)
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(img_data))
                ax.imshow(img)
                ax.set_title(f'{chart_name.replace("_", " ").title()}', fontsize=16, pad=20)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()

def generate_simple_text_report(feedback_text, student_data):
    """
    Generate a simple text report as fallback.
    """
    overall = student_data["overall_performance"]
    
    report = f"""
STUDENT PERFORMANCE ANALYSIS REPORT
Generated on: {datetime.now().strftime('%B %d, %Y')}

{'='*50}
PERFORMANCE SUMMARY
{'='*50}

Overall Score: {overall['studentTotalMarkScored']}/{overall['totalMarks']} ({overall['marksPercentage']:.1f}%)
Accuracy: {overall['studentOverallAccuracy']:.1f}%
Questions Attempted: {overall['studentTotalAttempted']}/{overall['totalQuestions']}
Time Utilization: {overall['timeUtilization']:.1f}%

{'='*50}
SUBJECT PERFORMANCE
{'='*50}

"""
    
    for subject in student_data["subject_performance"]:
        report += f"{subject['subjectName']}: {subject['accuracy']:.1f}% accuracy, {subject['totalAttempted']} attempted\n"
    
    report += f"""
{'='*50}
AI FEEDBACK
{'='*50}

{feedback_text}
"""
    
    return report.encode('utf-8')

# --- Streamlit Application ---
st.set_page_config(page_title="AI Student Performance Feedback", layout="wide", page_icon="📊")

st.title("🎓 AI-Powered Student Performance Feedback System")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 🚀 Advanced Features
    - **AI-Powered Analysis**: Personalized feedback using Google Gemini
    - **Comprehensive Visualizations**: Multiple chart types for deeper insights
    - **Professional PDF Reports**: Styled, downloadable reports
    - **Multi-level Analysis**: Overall, subject, chapter, and concept-wise performance
    """)

with col2:
    st.info("📁 Upload your test submission JSON file or use sample data to get started!")

# File upload section
st.markdown("### 📤 Upload Test Data")
uploaded_file = st.file_uploader("Choose a JSON file", type="json", help="Upload student test submission data in JSON format")

# Load and process data
json_data = None
if uploaded_file is not None:
    try:
        json_data = json.load(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON.")
else:
    if os.path.exists("data.json"):
        with open("data.json", "r") as f:
            json_data = json.load(f)
        st.info("ℹ️ Using sample data.json for demonstration.")

# Main application logic
if json_data:
    with st.spinner("🔄 Processing student data..."):
        student_data = parse_student_data(json_data)
    
    if student_data:
        # Display quick overview
        st.markdown("### 📊 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        overall = student_data["overall_performance"]
        
        with col1:
            st.metric("Overall Score", f"{overall['marksPercentage']:.1f}%")
        with col2:
            st.metric("Accuracy", f"{overall['studentOverallAccuracy']:.1f}%")
        with col3:
            st.metric("Attempted", f"{overall['attemptRate']:.1f}%")
        with col4:
            st.metric("Time Used", f"{overall['timeUtilization']:.1f}%")
        
        # Generate visualizations
        st.markdown("### 📈 Performance Visualizations")
        with st.spinner("Creating visualizations..."):
            charts = create_comprehensive_visualizations(student_data)
        
        # Display some charts in the UI
        col1, col2 = st.columns(2)
        with col1:
            if 'subject_radar' in charts:
                st.markdown("#### Subject Performance Radar")
                st.image(base64.b64decode(charts['subject_radar']))
        
        with col2:
            if 'time_accuracy' in charts:
                st.markdown("#### Time vs Accuracy")
                st.image(base64.b64decode(charts['time_accuracy']))
        
        # Chapter heatmap (full width)
        if 'chapter_heatmap' in charts:
            st.markdown("#### Chapter Performance Heatmap")
            st.image(base64.b64decode(charts['chapter_heatmap']))
        
        # Generate AI feedback
        st.markdown("### 🤖 AI-Generated Feedback")
        
        if st.button("🎯 Generate Personalized Feedback & PDF Report", type="primary"):
            with st.spinner("🧠 AI is analyzing performance and generating personalized feedback..."):
                try:
                    prompt = construct_advanced_feedback_prompt(student_data)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(prompt)
                    feedback_text = response.text
                    
                    # Display feedback
                    st.markdown("#### 📝 Personalized Feedback")
                    st.markdown(feedback_text)
                    
                    # Generate PDF with fallback methods
                    with st.spinner("📄 Generating PDF report..."):
                        pdf_bytes = None
                        pdf_filename = f"student_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Try ReportLab first (professional PDF)
                        if HAS_REPORTLAB:
                            try:
                                pdf_bytes = generate_pdf_report_reportlab(feedback_text, student_data, charts)
                                pdf_filename += ".pdf"
                                st.success("✅ Professional PDF generated using ReportLab!")
                            except Exception as e:
                                st.warning(f"ReportLab PDF generation failed: {e}")
                        
                        # Fallback to matplotlib PDF
                        if pdf_bytes is None and HAS_PDF_BACKEND:
                            try:
                                pdf_bytes = generate_pdf_report_matplotlib(feedback_text, student_data, charts)
                                pdf_filename += ".pdf"
                                st.success("✅ PDF generated using matplotlib!")
                            except Exception as e:
                                st.warning(f"Matplotlib PDF generation failed: {e}")
                        
                        # Final fallback to text report
                        if pdf_bytes is None:
                            pdf_bytes = generate_simple_text_report(feedback_text, student_data)
                            pdf_filename += ".txt"
                            st.info("ℹ️ Generated text report (PDF libraries not available)")
                        
                        # Download button
                        if pdf_bytes:
                            mime_type = "application/pdf" if pdf_filename.endswith('.pdf') else "text/plain"
                            st.download_button(
                                label="📥 Download Report",
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                mime=mime_type
                            )
                    
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating feedback: {e}")
        
        # Detailed data explorer
        with st.expander("🔍 Detailed Performance Data"):
            tab1, tab2, tab3 = st.tabs(["📚 Subjects", "📖 Chapters", "🎯 Concepts"])
            
            with tab1:
                st.dataframe(pd.DataFrame(student_data["subject_performance"]))
            
            with tab2:
                chapter_df = pd.DataFrame([
                    {
                        "Chapter": name,
                        "Accuracy": f"{data['accuracy']:.1f}%",
                        "Questions": data['totalQuestions'],
                        "Avg Time": f"{data['avgTimePerQuestion']:.1f}s"
                    }
                    for name, data in student_data["chapter_performance"].items()
                ])
                st.dataframe(chapter_df)
            
            with tab3:
                concept_df = pd.DataFrame([
                    {
                        "Concept": name,
                        "Mastery Level": data['masteryLevel'].title(),
                        "Accuracy": f"{data['accuracy']:.1f}%",
                        "Questions": data['totalQuestions']
                    }
                    for name, data in student_data["concept_performance"].items()
                ])
                st.dataframe(concept_df)
    
    else:
        st.error("❌ Could not process the JSON data. Please check the file format.")

else:
    st.warning("⚠️ Please upload a JSON file to get started.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Powered by Google Gemini AI • 📊 Built with Streamlit • 📄 Professional PDF Reports</p>
</div>
""", unsafe_allow_html=True)