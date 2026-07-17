import os
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
from tools import get_dataset_stats_tool, query_laptops_tool


def build_fallback_recommendation(user_profile: dict, error_message: str = None) -> str:
    """Generate a richer local recommendation report from the cleaned dataset when the LLM provider fails."""
    from data_manager import get_laptops_data

    rows = get_laptops_data()
    if not rows:
        return "The laptop database could not be loaded, so no local recommendations are available right now."

    df = pd.DataFrame(rows)
    if df.empty:
        return "The laptop database is empty, so no local recommendations are available right now."

    filtered_df = df.copy()

    budget = user_profile.get("budget")
    if budget is not None:
        try:
            budget_value = float(budget)
            filtered_df = filtered_df[filtered_df["Price"] <= budget_value]
        except (TypeError, ValueError):
            budget_value = None
    else:
        budget_value = None

    brand = user_profile.get("brand", "Any")
    if brand and str(brand).strip().lower() != "any":
        filtered_df = filtered_df[filtered_df["Brand"].astype(str).str.lower() == str(brand).strip().lower()]

    ram = user_profile.get("ram")
    if ram not in (None, "Any", "any"):
        try:
            filtered_df = filtered_df[filtered_df["RAM"] >= float(ram)]
        except (TypeError, ValueError):
            pass

    os_name = user_profile.get("os", "Any")
    if os_name and str(os_name).strip().lower() != "any":
        os_clean = str(os_name).strip().lower()
        if os_clean == "mac":
            os_clean = "macos"
        filtered_df = filtered_df[filtered_df["OS"].astype(str).str.lower() == os_clean]

    if filtered_df.empty:
        filtered_df = df.copy()

    filtered_df = filtered_df.sort_values(by=["Rating", "Price"], ascending=[False, True])
    top_matches = filtered_df.head(3)

    major = str(user_profile.get("major", "student") or "student").strip()
    major_lower = major.lower()

    if any(keyword in major_lower for keyword in ["computer science", "cs", "software", "engineering", "programming"]):
        workload = "coding, IDEs, virtual machines, and multitasking"
        fit_summary = "These picks are strong for software development and heavy browser/tab usage."
    elif any(keyword in major_lower for keyword in ["design", "graphic", "animation", "media"]):
        workload = "creative applications and visual work"
        fit_summary = "These picks are suitable for design and content creation workflows."
    elif any(keyword in major_lower for keyword in ["business", "finance", "marketing"]):
        workload = "office apps, presentations, and daily productivity"
        fit_summary = "These picks are practical for general productivity and portability."
    else:
        workload = "general study, research, and everyday productivity"
        fit_summary = "These picks balance price, performance, and reliability for day-to-day academic use."

    intro = "The AI recommendation service was unavailable, so I used the local laptop database to suggest the closest matches."
    if error_message:
        intro += f" Error: {error_message}"

    lines = [intro, "", "## Personalized Laptop Recommendation Report", "", f"**Profile:** {major}", f"**Budget target:** ₹{budget_value:,.0f} if available" if budget_value is not None else "**Budget target:** Flexible", f"**Use case:** {workload}", "", fit_summary, ""]

    for idx, row in top_matches.iterrows():
        price = int(row["Price"])
        ram = int(row["RAM"])
        storage = int(row["SSD_Storage"])
        rating = float(row["Rating"])
        brand = str(row["Brand"])
        processor = str(row["Processor"])
        os_name = str(row["OS"])

        reasons = []
        if ram >= 16:
            reasons.append("high RAM for multitasking and heavier apps")
        if storage >= 512:
            reasons.append("ample SSD storage for files, projects, and software")
        if rating >= 4.5:
            reasons.append("strong user rating")
        if processor in ["Intel", "AMD"]:
            reasons.append("solid general-purpose processor performance")

        pros = []
        if ram >= 16:
            pros.append("Excellent for coding, multitasking, and IDEs")
        if storage >= 512:
            pros.append("Enough space for study materials, applications, and media")
        if rating >= 4.5:
            pros.append("Good overall user satisfaction")

        cons = []
        if budget_value is not None and price > budget_value:
            cons.append("Price is above the stated budget")
        if ram < 16:
            cons.append("Lower RAM may feel limiting for heavier workloads")
        if storage < 512:
            cons.append("Smaller SSD may fill up quickly over time")

        if not pros:
            pros.append("Balanced price-to-performance option")
        if not cons:
            cons.append("No major trade-off listed in the dataset")

        lines.append(f"### {idx + 1}. {brand} — ₹{price:,}")
        lines.append(f"- **Why it fits:** {brand} offers {ram}GB RAM, {storage}GB storage, {processor} processing, and {os_name} — a good fit for {workload}.")
        lines.append(f"- **Pros:** {', '.join(pros)}.")
        lines.append(f"- **Cons:** {', '.join(cons)}.")
        lines.append(f"- **Best for:** {major} students who want a practical balance of price and reliability.")
        lines.append("")

    lines.append("### Side-by-side comparison")
    lines.append("")
    lines.append("| Laptop | Price | RAM | Storage | Processor | OS | Rating | Best fit |")
    lines.append("|---|---:|---:|---:|---|---|---:|---|")
    for _, row in top_matches.iterrows():
        lines.append(f"| {row['Brand']} | ₹{int(row['Price']):,} | {int(row['RAM'])}GB | {int(row['SSD_Storage'])}GB | {row['Processor']} | {row['OS']} | {row['Rating']}/5 | {major} / {workload} |")
    lines.append("")
    lines.append("### Final recommendation")
    best_row = top_matches.iloc[0]
    lines.append(f"**Best overall pick:** {best_row['Brand']} at ₹{int(best_row['Price']):,} because it gives the strongest mix of price, RAM, storage, and rating from the available local data.")
    return "\n".join(lines)


def create_laptop_crew(api_key: str, provider: str, user_profile: dict) -> Crew:
    """
    Create a Crew to analyze user requirements, query the laptop database,
    and generate custom recommendations.
    
    api_key: User provided LLM API Key (OpenAI or Gemini)
    provider: 'openai' or 'gemini'
    user_profile: dict containing budget, major, RAM, etc.
    """
    # Set default keys from env if frontend didn't supply them
    if not api_key:
        if provider == 'gemini':
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == 'openai':
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider in ('kimi', 'nvidia_kimi'):
            api_key = os.environ.get("NVIDIA_API_KEY") 
    if not api_key:
        raise ValueError(f"API Key for {provider} not found. Please provide it in the UI or environment.")

    # Initialize the appropriate LLM
    if provider == 'gemini':
        # Clean model name for crewai LLM class
        llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.3
        )
    elif provider == 'openai':
        llm = LLM(
            model="openai/gpt-4o-mini",
            api_key=api_key,
            temperature=0.3
        )
    elif provider in ('kimi', 'nvidia_kimi'):
        # NVIDIA API gateway for Moonshot AI Kimi model
        llm = LLM(
            model="openai/moonshotai/kimi-k2.6",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            temperature=0.3
        )
    else:
        # Fallback default
        llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.3
        )

    # Convert user profile to descriptive string for agents
    profile_desc = f"""
    - **Academic Major / Workload**: {user_profile.get('major', 'Student')}
    - **Price Budget**: Max {user_profile.get('budget', 'Any')} INR
    - **Minimum RAM**: {user_profile.get('ram', 'Any')} GB
    - **Preferred Brand**: {user_profile.get('brand', 'Any')}
    - **Preferred OS**: {user_profile.get('os', 'Any')}
    - **Specific Details/Preferences**: {user_profile.get('details', 'None')}
    """

    # Agent 1: Data Specialist Agent
    data_specialist = Agent(
        role="Laptop Database Specialist",
        goal="Search the database using filters to find matching laptops that fit the user's budget and technical constraints.",
        backstory="""You are a meticulous data engineer. Your job is to query the laptop database
        using custom search tools. You look for laptops that fall within the user's budget and meet
        their basic RAM, storage, brand, and OS needs. You gather evidence and output exact product data,
        ensuring that no laptop specs or prices are made up. You always double-check the raw rows before presenting them.""",
        tools=[get_dataset_stats_tool, query_laptops_tool],
        llm=llm,
        verbose=True
    )

    # Agent 2: Recommender Agent
    recommender = Agent(
        role="Academic & Professional Hardware Advisor",
        goal="Select the top 3 best matching laptops from the database search results, and explain how their specifications suit the user's academic major or professional workload.",
        backstory="""You are an expert computing systems consultant and academic advisor.
        You understand hardware inside out. You know that a Computer Science student needs a fast CPU,
        at least 16GB RAM for running Docker/VMs/compilers, and an SSD; a Graphic Designer needs a premium screen
        and strong processor; a business student needs long battery life and portability.
        Your goal is to explain *why* the recommended specs fit the customer's specific studies/workload,
        breaking down the trade-offs (Pros & Cons) of each recommendation in a friendly, persuasive tone.""",
        llm=llm,
        verbose=True
    )

    # Task 1: Search the dataset
    search_task = Task(
        description=f"""
        1. Examine the user profile criteria:
        {profile_desc}
        
        2. First, call the statistics tool to check the dataset stats (price ranges, brand counts) if needed to understand what is available.
        3. Call the 'Query Laptops Dataset' tool with filters representing the user's budget, preferred brand, RAM, and OS to find matching laptops.
        4. If no laptops match the exact combination, relax some filters (e.g. increase the budget slightly, or search for other brands with similar RAM) to find the closest matches.
        5. Compile a list of the matching laptops, keeping their exact price, RAM, Storage, Processor, OS, and rating.
        """,
        expected_output="A markdown table containing the matching laptops retrieved from the database, along with their specifications and prices. Do not invent models or specs.",
        agent=data_specialist
    )

    # Task 2: Formulate recommendations
    recommendation_task = Task(
        description=f"""
        1. Analyze the matching laptops table from the search task.
        2. Choose the top 3 best matching options for the user's profile:
        {profile_desc}
        
        3. Write a comprehensive, personalized recommendation report.
        4. For each laptop, explain:
           - Why the hardware specs (RAM, Storage, Processor, OS) fit their academic major or workload.
           - Specific benefits for their profile (e.g. 'Since you are in Computer Science, 16GB RAM allows you to run virtual machines and IDEs smoothly...').
           - Pros and Cons (e.g. price vs. rating, color, brand reputation).
           - How it fits their financial constraints.
        5. Formulate a final comparison table comparing the recommended models.
        6. Structure the report beautifully in Markdown with sections, clear bullet points, bold highlights, and clean typography.
        
        Do not make up any specifications, ratings, or prices that are not present in the database search results.
        """,
        expected_output="A beautiful, detailed markdown report listing the top 3 laptop recommendations, including academic/professional justifications, pros/cons, and a final summary comparison table.",
        agent=recommender
    )

    # Combine into a Crew
    crew = Crew(
        agents=[data_specialist, recommender],
        tasks=[search_task, recommendation_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew
