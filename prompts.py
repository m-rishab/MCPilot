"""
prompts.py - Contains system prompts and templating for the MCP agent
"""

# System prompt for the agent

SYSTEM_PROMPT = """You are an AI assistant with access to various tools.

For web browsing tasks:
- First verbalize: "Let me navigate to this website for you using my browser tools"
- Use the playwright server with browser_navigate action
- After each step, explain what you're doing: "Now I'll check what's on the page"
- Use browser_snapshot to observe the page content
- Narrate your exploration: "I can see... Let me look for..."

For information searches:
- Say: "I'll search for this information using my search tools"
- Use the duckduckgo-search server with duckduckgo_web_search action
- Share your process: "Let me analyze these search results for you"
- Present findings conversationally: "I found that..."

For weather requests:
- Begin with: "I'll check the current weather using my weather tools"
- Use the weather_forecast server with appropriate actions
- Explain your process: "First getting current conditions, then checking the forecast"
- Present weather information as a helpful report

For accommodation searches:
- Start with: "I'll use my booking tools to find accommodations"
- Use the appropriate server (airbnb/hotel) with search actions
- Narrate your search: "Searching for places in this location..."
- Present options conversationally

When learning new information:
- Acknowledge: "I'm learning about this now..."
- Ask clarifying questions when needed
- Confirm understanding: "Based on what I'm seeing..."

Always take one action at a time, describe what you're doing, and maintain a conversational tone throughout.
"""

# Weather request template

def get_weather_prompt(location):
    return f"""I'll check the weather in {location} for you.

First, let me use my weather tool to get current conditions:
"I'll use the weather_forecast server with get_current_weather action to check what's happening right now."

{{
 "location": "{location}",
 "timezone_offset": 5.5
}}

Now I need to get the forecast:
"Let me also get the forecast to give you a complete picture."

Use get_weather action with the same parameters.

After receiving the data:
"Here's what I found about the weather in {location}..."

I'll include:
- Current temperature and conditions
- Humidity and wind information
- High and low temperatures for today
- Forecast for the next few days
"""

# Search request template

def get_search_prompt(query):
    return f"""I'll search for information about "{query}" for you.

"Let me use my search tools to find the most relevant information."

I'll use the duckduckgo-search server with action 'duckduckgo_web_search':

{{
 "query": "{query}"
}}

"Now I'm analyzing the search results to find the most helpful information for you."

I'll share what I learn, including specific details, names, dates, and key facts."""

# Web navigation template for simple navigation

def get_navigation_prompt(url):
    return f"""I'll navigate to {url} for you.

"Let me use my browser tools to visit this website."

I'll use the playwright server with browser_navigate action:

{{
 "url": "{url}"
}}

"Now I'll take a snapshot to see what's on the page."

I'll use browser_snapshot to observe the content.

"Here's what I can see on the website..."

I'll describe the main content, layout, and any important information I find."""

# Web navigation template with search

def get_navigation_with_search_prompt(url, search_term):
    return f"""I'll navigate to {url} and search for "{search_term}" for you.

"First, let me visit the website using my browser tools."

I'll use the playwright server with browser_navigate action:

{{
 "url": "{url}"
}}

"Now I'll take a snapshot to see what's on the page."

I'll use browser_snapshot to observe the content.

"I'm looking for the search field on this page."

I'll use browser_type action to input "{search_term}" into the search field:

"Now I'll submit the search query."

I'll use browser_click to submit or press Enter.

"Let me check the search results for you."

I'll use browser_snapshot again to see the results.

"Here's what I found about {search_term} on this website..."

I'll describe the search results in detail."""

# Airbnb search template

def get_airbnb_prompt(location):
    return f"""I'll find accommodations in {location} for you.

"Let me use my Airbnb tools to search for available places."

I'll use the airbnb server with the appropriate search action:

{{
 "location": "{location}"
}}

"I'm now reviewing the available listings to find the best options."

I'll share details about:
- Property names and types
- Pricing information
- Key features and amenities
- Guest ratings and reviews
- Location highlights

"Here are the best accommodation options I found in {location}..."

I'll present the most relevant listings based on quality and value."""

# Hotel search template

def get_hotel_search_prompt(location, website=None):
    """Generate prompt for hotel search on a specific website"""
    if website:
        if "makemytrip" in website.lower():
            return f"""I'll find hotel information in {location} on {website} for you.
            
"Let me navigate to {website} using my browser tools."

I'll follow these steps:
1. "First, I'll visit the website." - Use browser_navigate to go to {website}
2. "Now I'll look for the Hotels section." - Find and click on the Hotels tab
3. "I'll search for {location}." - Enter the location in the search field
4. "Submitting the search now." - Submit the search query
5. "Waiting for results to load." - Allow the page to load completely
6. "I'm analyzing the top hotel options." - Review the search results
7. "Let me gather details about the best options." - Extract information about:
   - Hotel names
   - Star ratings
   - Price ranges
   - Key amenities
   - User ratings

"Here are the best hotel options I found in {location}..."

I'll provide a concise summary of the top 3-5 hotels with all relevant details.
"""
        else:
            return f"""I'll find hotel information in {location} on {website} for you.

"Let me navigate to {website} using my browser tools."

First, I'll visit the website:
{{
 "url": "{website}"
}}

"Now I'll look for ways to search for hotels in {location}."

I'll locate the search functionality and input the location.

"I'm reviewing the available hotels to find the best options."

I'll gather information about:
- Hotel names and ratings
- Price ranges and deals
- Key amenities and features
- Guest reviews and satisfaction scores

"Here are the best hotel options I found in {location}..."

I'll present the most relevant options based on quality, value, and amenities.
"""
    else:
        return f"""I'll search for top-rated hotels in {location} for you.

"Let me use my search tools to find the best hotels."

I'll search for: "best hotels in {location} prices reviews ratings"

"I'm analyzing the search results to find reliable information."

I'll gather details about:
- Highly-rated hotels
- Price ranges
- Key amenities
- Location advantages
- Guest experiences

"Here are the top hotel options in {location} based on my search..."

I'll present a summary of the best options I've found."""

# Clarification responses

CLARIFICATION_SEARCH_OR_NAVIGATE = """I notice you've mentioned both searching and a specific website. I want to make sure I understand what you'd prefer:

1. Would you like me to navigate directly to the website and explore what's there?
2. Or would you prefer I search for information about the website first?

Let me know which approach would be more helpful for you."""

def get_research_paper_clarification(topic):
    return f"""I'd be happy to find research papers on {topic} for you. To better meet your needs, could you clarify what type of papers you're most interested in:

1. Most cited papers - These would be influential works with high citation counts
2. Recent breakthrough papers - The latest developments and findings in this field
3. Foundational papers - Classic works that established key concepts in this area

Once I know your preference, I can tailor my search accordingly."""