import os
import json
import asyncio
import gradio as gr
import requests
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
from importlib.metadata import version, PackageNotFoundError
from langchain_core.messages import SystemMessage
from typing import Dict, List, Any
import prompts  # Import the prompts file

# ========== ENVIRONMENT ==========
def load_environment():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    weather_key = os.getenv("WEATHER_API_KEY")
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["WEATHER_API_KEY"] = weather_key
    print(f"GROQ API Key loaded: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"Weather API Key loaded: {'✅ Set' if weather_key else '❌ Missing'}")
    
    # Test the weather API key if present
    if weather_key:
        test_weather_api()

def test_weather_api():
    try:
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            print("❌ No weather API key available to test")
            return
            
        test_url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={weather_key}&units=metric"
        response = requests.get(test_url)
        if response.status_code == 200:
            print("✅ Weather API key is valid and working")
        else:
            print(f"❌ Weather API test failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Weather API test exception: {e}")

# ========== INITIALIZE AGENT ==========
async def get_agent():
    try:
        config_path = os.path.abspath("browser_mcp.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} not found. Please create it with valid MCP servers.")

        with open(config_path, "r") as f:
            config_data = json.load(f)
        print(f"Loaded MCP Config")

        # Add unique profile directory for Playwright
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        # Update the profile directory with a unique name
        if "playwright" in config_data["mcpServers"]:
            for i, arg in enumerate(config_data["mcpServers"]["playwright"]["options"]["args"]):
                if "--user-data-dir=" in arg:
                    config_data["mcpServers"]["playwright"]["options"]["args"][i] = f"--user-data-dir=/tmp/playwright-profile-{unique_id}"
                    
        # Save the updated config
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        try:
            client = MCPClient.from_config_file(config_path)
            print("✅ MCPClient created from config file")
        except Exception as e1:
            print(f"❌ Failed to create client from config file: {e1}")
            try:
                client = MCPClient.from_dict(config_data)
                print("✅ MCPClient created from dict")
            except Exception as e2:
                print(f"❌ Failed to create client from dict: {e2}")
                raise ValueError("❌ Could not initialize MCPClient from either method")

        server_names = client.get_server_names()
        print(f"Servers found: {server_names}")
        if not server_names:
            raise ValueError("❌ No MCP servers found in the config")

        # Create sessions for all available servers
        for server_name in server_names:
            try:
                await client.create_session(server_name)
                print(f"✅ Session created for {server_name}")
            except Exception as e:
                print(f"⚠️ Could not create session for {server_name}: {e}")

        # Use the system prompt from prompts.py
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0.2)
        agent = MCPAgent(llm=llm, client=client, max_steps=15, memory_enabled=True) 
        agent.set_system_message(prompts.SYSTEM_PROMPT)

        return agent

    except Exception as e:
        print(f"❌ Agent setup failed: {e}")
        import traceback
        print(traceback.format_exc())
        raise

# ========== HELPER FUNCTIONS ==========
def extract_url_from_message(message: str) -> str:
    """Extract and validate URL from a user message"""
    if not message:
        return None
        
    url_pattern = re.compile(r'https?://\S+|(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}')
    urls = url_pattern.findall(message)
    
    if urls:
        target_url = urls[0]
        if not target_url.startswith('http'):
            target_url = 'https://' + target_url
        return target_url
    return None

def find_url_in_history(history):
    """Find URL from conversation history"""
    if not history:
        return None
        
    # Go through history in reverse to find most recent URL
    for msg in reversed(history):
        if msg.get("role") == "user":
            url = extract_url_from_message(msg.get("content", ""))
            if url:
                return url
    return None

def extract_location_from_message(message: str, default="New York") -> str:
    """Extract location information from a message"""
    # Clean up the message to focus on location
    # First remove common action phrases
    for prefix in ["tell me", "what is", "how is", "show me", "get"]:
        message = re.sub(fr'\b{prefix}\b', '', message, flags=re.IGNORECASE)
    
    cleaned_msg = message.lower()
    for term in ["weather", "forecast", "temperature", "hotel", "hotels", "airbnb", 
                 "accommodation", "in", "at", "near", "for", "the", "of", 
                 "search", "find", "lookup", "about", "best"]:
        cleaned_msg = cleaned_msg.replace(term, " ").strip()
    
    # Remove extra spaces
    cleaned_msg = re.sub(r'\s+', ' ', cleaned_msg).strip()
    
    # If we have something left, use it as location, otherwise use default
    return cleaned_msg if cleaned_msg else default

def extract_search_term(message: str) -> str:
    """Extract search term from navigation + search message"""
    search_match = re.search(r"search(?:\s+for)?\s+(.+?)(?:$|\son|\sin)", message.lower())
    if search_match:
        return search_match.group(1).strip()
    return None

def handle_compound_navigation_search(message):
    """Handle requests that combine navigation and search"""
    # Look for patterns like "open X and search Y" or "navigate to X then search for Y"
    compound_pattern = re.search(r"(?:open|navigate to|go to|visit)\s+([^\s]+)(?:\s+and|\s+then|\s*,\s*)?\s+search(?:\s+for)?\s+(.+?)(?:$|\son|\sin)", message.lower())
    
    if not compound_pattern:
        return None, None
    
    nav_url = compound_pattern.group(1)
    if not nav_url.startswith('http'):
        nav_url = 'https://' + nav_url
    
    search_term = compound_pattern.group(2).strip()
    
    return nav_url, search_term

def is_research_papers_request(message: str) -> tuple:
    """Check if the message is asking for research papers and extract the topic"""
    if "research papers" in message.lower():
        topic_match = re.search(r"(?:on|about)\s+(.+?)(?:$|\?)", message.lower())
        if topic_match:
            return True, topic_match.group(1).strip()
    return False, None

async def handle_weather_request(agent, message):
    """Handle weather-related requests by directly using OpenWeatherMap API"""
    location = extract_location_from_message(message)
    print(f"Extracted location for weather: '{location}'")
    
    try:
        # Get the weather API key
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            return "Weather API key is not available. Please check your environment setup."
            
        # Make direct API call to OpenWeatherMap
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_key}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Could not retrieve weather data for {location}. Please check the location name and try again."
        
        # Parse the weather data
        weather_data = response.json()
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        
        # Format the response
        weather_response = f"Current weather in {location}: {temp}°C, {description}\n"
        weather_response += f"Humidity: {humidity}%, Wind: {wind_speed} m/s"
        
        # Get forecast if available
        try:
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={weather_key}&units=metric&cnt=8"
            forecast_response = requests.get(forecast_url)
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                forecast_items = forecast_data['list']
                
                if forecast_items:
                    weather_response += "\n\nForecast for the next 24 hours:\n"
                    for item in forecast_items[:3]:  # Just show first 3 periods
                        time = item['dt_txt'].split(' ')[1][:5]  # Extract just the time HH:MM
                        temp = item['main']['temp']
                        desc = item['weather'][0]['description']
                        weather_response += f"• {time}: {temp}°C, {desc}\n"
        except Exception as e:
            print(f"Forecast error: {e}")
            # Don't add forecast to response if there's an error
            
        return weather_response
        
    except Exception as e:
        print(f"Weather API error: {e}")
        import traceback
        print(traceback.format_exc())
        return f"I encountered an error getting weather data for {location}. Please try again."

async def handle_search_request(agent, message):
    """Handle search-related requests"""
    # Extract search query by removing common prefixes
    query = message
    for prefix in ["search for", "search", "find", "lookup", "information on", "tell me about"]:
        if query.lower().startswith(prefix):
            query = query[len(prefix):].strip()
            break
    
    try:
        # Temporarily lower temperature for more precise search results
        original_temp = agent.llm.temperature
        agent.llm.temperature = 0.1
        response = await agent.run(prompts.get_search_prompt(query))
        agent.llm.temperature = original_temp
        
        # Clean up the response to extract just the search results
        if "Final Answer:" in response:
            response = response.split("Final Answer:")[1].strip()
        
        return response
    except Exception as e:
        print(f"Search API error: {e}")
        return f"I encountered an error searching for '{query}'. Please try again."

async def handle_navigation_request(agent, message):
    """Handle web navigation requests"""
    target_url = extract_url_from_message(message)
    if not target_url:
        return "I couldn't identify a valid website URL in your message. Please specify a website to visit."
    
    print(f"Navigating to: {target_url}")
    
    # Check if there's also a search intent
    search_term = extract_search_term(message)
    
    try:
        if search_term:
            print(f"Will search for '{search_term}' after navigating")
            response = await agent.run(prompts.get_navigation_with_search_prompt(target_url, search_term))
        else:
            response = await agent.run(prompts.get_navigation_prompt(target_url))
            
        # Clean up the response
        if "Final Answer:" in response:
            response = response.split("Final Answer:")[1].strip()
        
        return response
    except Exception as e:
        print(f"Navigation error: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Provide a simplified response if there's an error
        return f"I attempted to navigate to {target_url}, but encountered an error. This could be due to website restrictions or connectivity issues."

async def handle_airbnb_request(agent, message):
    """Handle Airbnb accommodation searches"""
    location = extract_location_from_message(message)
    print(f"Extracted location for Airbnb: '{location}'")
    
    try:
        response = await agent.run(prompts.get_airbnb_prompt(location))
        
        # Clean up the response
        if "Final Answer:" in response:
            response = response.split("Final Answer:")[1].strip()
            
        return response
    except Exception as e:
        print(f"Airbnb API error: {e}")
        return f"I encountered an error searching for accommodations in {location}. Please try again."

async def handle_hotel_search_request(agent, message):
    """Handle hotel search requests on specific travel websites"""
    # Extract location
    location_pattern = re.search(r"hotels? in ([a-zA-Z\s]+)", message.lower())
    location = location_pattern.group(1).strip() if location_pattern else "Faridabad"
    
    # Extract website if mentioned
    website = None
    site_pattern = re.search(r"(?:from|on|at|using) ([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,})", message.lower())
    if site_pattern:
        website = site_pattern.group(1).strip()
        if not website.startswith("http"):
            website = "https://" + website
    
    print(f"Searching for hotels in {location} on {website if website else 'travel websites'}")
    
    try:
        if website:
            if "makemytrip" in website:
                # Specific handling for MakeMyTrip
                search_prompt = f"""
                Navigate to {website} and search for hotels in {location}.
                Follow these steps:
                1. Navigate to the website
                2. Look for the "Hotels" tab or section and click on it
                3. Enter "{location}" in the destination/location search field
                4. Submit the search or click on search button
                5. Wait for the results to load
                6. Summarize the top 3-5 hotel options with their ratings and prices
                """
            else:
                # Generic handling for other travel sites
                search_prompt = f"""
                Navigate to {website} and search for hotels in {location}.
                Try to find and list the best available options with their ratings and prices.
                """
        else:
            # If no specific site mentioned, use search
            search_prompt = f"Search for best hotels in {location}"
            return await handle_search_request(agent, search_prompt)
        
        response = await agent.run(search_prompt)
        
        # Clean up the response
        if "Final Answer:" in response:
            response = response.split("Final Answer:")[1].strip()
            
        return response
    except Exception as e:
        print(f"Hotel search error: {e}")
        import traceback
        print(traceback.format_exc())
        return f"I encountered an error searching for hotels in {location}. Please try again with a more specific request."

# ========== CHAT HANDLER ==========
agent = None

async def chat_with_agent(message, history):
    global agent

    if not agent:
        try:
            print("Initializing agent...")
            agent = await get_agent()
            print("✅ Agent initialized")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            print(traceback.format_exc())
            return f"❌ Agent initialization failed: {str(e)}"

    try:
        print(f"User message: {message}")
        
        # Check what servers are available
        available_servers = agent.client.get_server_names() if agent and hasattr(agent, "client") else []
        print(f"Available servers: {available_servers}")
        
        # Check for direct navigation commands first
        if any(term in message.lower() for term in ["navigate to", "go to", "open", "visit"]):
            url = extract_url_from_message(message)
            if not url:
                url = find_url_in_history(history)
                
            if url and "playwright" in available_servers:
                print(f"Direct navigation request to: {url}")
                return await handle_navigation_request(agent, f"navigate to {url}")
        
        # Handle compound navigation and search requests
        nav_url, search_term = handle_compound_navigation_search(message)
        if nav_url and "playwright" in available_servers:
            if search_term:
                print(f"Compound request: navigate to {nav_url} and search for {search_term}")
                return await handle_navigation_request(agent, f"navigate to {nav_url} and search for {search_term}")
            else:
                print(f"Navigation request extracted from compound command: {nav_url}")
                return await handle_navigation_request(agent, f"navigate to {nav_url}")
        
        # Check for clarification opportunities
        
        # 1. Combined search and navigation request
        if "search" in message.lower() and extract_url_from_message(message):
            # This is an ambiguous request - ask for clarification
            url = extract_url_from_message(message)
            print(f"Detected ambiguous search/navigate request for URL: {url}")
            return prompts.CLARIFICATION_SEARCH_OR_NAVIGATE
        
        # 2. Research papers request
        is_research_req, topic = is_research_papers_request(message)
        if is_research_req and topic:
            print(f"Detected research papers request on topic: {topic}")
            return prompts.get_research_paper_clarification(topic)
        
        # Handle responses to previous clarification questions
        if any(option in message.lower() for option in ["option 1", "option 2", "option 3", "#1", "#2", "#3", "choice 1", "choice 2", "choice 3", "1", "2", "3"]):
            # Extract context from previous messages
            context = ""
            for msg in history:
                if msg.get("role") == "assistant":
                    context = msg.get("content", "")
                    if "option" in context.lower() or "prefer" in context.lower():
                        break
            
            # Process the chosen option
            if "navigate directly" in context or "search for information" in context:
                url = extract_url_from_message(context)
                
                # If no URL in context, look through history
                if not url:
                    url = find_url_in_history(history)
                
                if url:
                    if any(x in message.lower() for x in ["option 1", "#1", "choice 1", "1", "navigate", "first"]):
                        print(f"User chose to navigate to: {url}")
                        response = await handle_navigation_request(agent, f"navigate to {url}")
                    else:
                        print(f"User chose to search for: {url}")
                        response = await handle_search_request(agent, f"search for {url}")
                    return response
            
            elif "research papers" in context:
                topic_match = re.search(r"research papers on (.+?)\.", context)
                topic = topic_match.group(1) if topic_match else "the topic"
                
                if any(x in message.lower() for x in ["option 1", "#1", "choice 1", "1", "most cited"]):
                    query = f"most cited research papers on {topic}"
                elif any(x in message.lower() for x in ["option 2", "#2", "choice 2", "2", "recent"]):
                    query = f"recent breakthrough research papers on {topic}"
                else:
                    query = f"foundational research papers on {topic}"
                
                print(f"User chose research papers option: {query}")
                response = await handle_search_request(agent, query)
                return response
        
        # Route the request based on content
        if any(w in message.lower() for w in ["weather", "temperature", "humidity", "forecast"]):
            # Direct API call to OpenWeatherMap instead of relying on MCP weather_forecast server
            response = await handle_weather_request(agent, message)
        
        # Hotel search handling - needs to come before general navigation to catch hotel-specific queries
        elif any(w in message.lower() for w in ["hotel", "hotels", "stay", "resort", "accommodation"]):
            if "playwright" in available_servers:
                response = await handle_hotel_search_request(agent, message)
            else:
                response = "I'm sorry, but the browser service needed for hotel searches is currently unavailable."
                
        elif any(term in message.lower() for term in ["open", "visit", "go to", "navigate", "browse"]) or \
             extract_url_from_message(message):
            if "playwright" in available_servers:
                response = await handle_navigation_request(agent, message)
            else:
                response = "I'm sorry, but the browser service is currently unavailable."
                
        elif any(w in message.lower() for w in ["search", "find", "lookup", "information on"]):
            if "duckduckgo-search" in available_servers:
                response = await handle_search_request(agent, message)
            else:
                response = "I'm sorry, but search tools are currently unavailable."
                
        elif any(w in message.lower() for w in ["airbnb", "place to stay"]):
            if "airbnb" in available_servers:
                response = await handle_airbnb_request(agent, message)
            else:
                response = "I'm sorry, but the Airbnb service is currently unavailable."
        
        else:
            # For general conversation, use the agent directly
            response = await agent.run(message)

        # Clean up the response
        if "Action:" in response and "Action Input:" in response:
            final_answer_idx = response.find("Final Answer:")
            if final_answer_idx != -1:
                response = response[final_answer_idx + len("Final Answer:"):].strip()
            else:
                # If no final answer, look for the last observation
                observation_idx = response.rfind("Observation:")
                if observation_idx != -1:
                    last_part = response[observation_idx + len("Observation:"):]
                    next_section = min(
                        last_part.find("Thought:") if last_part.find("Thought:") != -1 else float('inf'),
                        last_part.find("Action:") if last_part.find("Action:") != -1 else float('inf')
                    )
                    if next_section != float('inf'):
                        response = last_part[:next_section].strip()
                    else:
                        response = last_part.strip()
        
        # Handle error cases
        if "Agent stopped due to an error" in response or "Agent stopped after reaching" in response:
            # Extract only the essential information from the error message
            error_match = re.search(r"Error: (.+?)(?:\n|$)", response)
            error_msg = error_match.group(1) if error_match else "Unknown error occurred"
            response = f"I encountered an issue: {error_msg}\n\nLet me know if you'd like me to try a different approach."
        
        print(f"Final response: {response}")
        return response

    except Exception as e:
        print(f"❌ Error in chat_with_agent: {e}")
        import traceback
        print(traceback.format_exc())
        return f"I encountered an unexpected error. Please try again with a simpler request."

# ========== GRADIO UI ==========
with gr.Blocks(title="🧠 MCP Chatbot with Gradio") as demo:
    gr.Markdown("## 🤖 Hyper Agent\nChat with tools like Weather, Search, and Web Browsing")

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Chat History", 
            render=True, 
            type="messages",  # Use the modern messages format
            height=600,
            elem_id="chatbot"
        )

    user_input = gr.Textbox(
        label="Your message", 
        placeholder="Type here and press Enter...",
        container=True
    )

    def user_submit(msg, history):
        if not msg.strip():
            return history, ""
        history.append({"role": "user", "content": msg})
        try:
            response = asyncio.run(chat_with_agent(msg, history))
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            history.append({"role": "assistant", "content": f"❌ System error: {str(e)}"})
        return history, ""

    user_input.submit(fn=user_submit, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    # Add clear button
    clear_btn = gr.Button("Clear Chat")
    def clear_chat():
        global agent
        if agent:
            agent.clear_memory()  # Clear agent memory when clearing chat
        return [], ""
    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, user_input])

# ========== MAIN ==========
if __name__ == "__main__":
    load_environment()
    print("🚀 Starting Gradio...")
    demo.launch(share=True)