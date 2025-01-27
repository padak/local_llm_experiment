#!/usr/bin/env python3
import json
import requests
import platform
import os
import signal
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import openai  # Added for OpenRouter integration

# Load environment variables from .env file
load_dotenv()

console = Console()

def signal_handler(sig, frame):
    console.print("\n[yellow]Caught CTRL+C, exiting gracefully...[/yellow]")
    sys.exit(0)

# Set up signal handler for CTRL+C
signal.signal(signal.SIGINT, signal_handler)

AVAILABLE_MODELS = {
    # Ollama models (local)
    "1": {"name": "deepseek-r1:1.5b", "backend": "ollama", "display": "DeepSeek R1 1.5B"},
    "2": {"name": "deepseek-r1:7b", "backend": "ollama", "display": "DeepSeek R1 7B"},
    "3": {"name": "mistral:7b", "backend": "ollama", "display": "Mistral 7B"},
    
    # vLLM models (local)
    "4": {"name": "TheBloke/Mistral-7B-v0.1-GGUF", "backend": "vllm", "display": "Mistral 7B GGUF"},
    
    # OpenRouter models with specific providers
    "5": {
        "name": "deepseek/deepseek-r1",
        "backend": "cloud",
        "provider": "openrouter",
        "provider_name": "avian",  # Specify Avian as provider
        "display": "DeepSeek R1 (via Avian)"
    },
    "6": {
        "name": "deepseek/deepseek-r1",  # Same model as 5, different provider
        "backend": "cloud",
        "provider": "openrouter",
        "provider_name": "novita",  # Specify NovitaAI as provider
        "display": "DeepSeek R1 (via NovitaAI)"
    },
    "7": {
        "name": "deepseek/deepseek-r1",  # Same model, direct from DeepSeek
        "backend": "cloud",
        "provider": "openrouter",
        "provider_name": "deepseek",  # Specify DeepSeek as provider
        "display": "DeepSeek R1 (via DeepSeek)"
    }
}

REASONING_PROMPT = "You are a helpful AI assistant. Focus on providing clear step-by-step reasoning for your answers. Break down complex problems into logical steps and explain your thought process."

def is_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def chat_with_ollama(prompt, model_name):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Request interrupted by user[/yellow]")
        return None

def chat_with_openrouter(prompt, model_name, provider_name=None, system_prompt=None):
    # Configure OpenAI client for OpenRouter
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": os.getenv('GITHUB_REPO_URL', 'https://github.com/yourusername/local-llm-deepseek'),
            "X-Title": "Local LLM DeepSeek Chat"
        }
    )
    
    try:
        # Prepare messages with provider routing if specified
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_message = {"role": "user", "content": prompt}
        if provider_name:
            user_message["name"] = f"via_{provider_name}"  # Add provider routing
        messages.append(user_message)
        
        # Stream the response
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=512
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                console.print(content, end="")
        
        console.print()  # New line after streaming
        return full_response
            
    except openai.APITimeoutError:
        console.print("\n[red]Error: Request timed out. Please try again.[/red]")
        return None
    except openai.APIError as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Request interrupted by user[/yellow]")
        return None

def chat_with_vllm(prompt, model_name, llm_instance):
    try:
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            stop=["</s>", "User:", "Assistant:"]
        )
        outputs = llm_instance.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Request interrupted by user[/yellow]")
        return None

def chat_with_avian(prompt, model_name):
    url = "https://api.avian.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('AVIAN_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True
    }
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            
                            chunk = json.loads(json_str)
                            if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                content = chunk['choices'][0]['delta']['content']
                                full_response += content
                                console.print(content, end="")
                    
                    except json.JSONDecodeError:
                        continue
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Request interrupted by user[/yellow]")
                        return None
            
            console.print()
            return full_response
            
    except requests.exceptions.Timeout:
        console.print("\n[red]Error: Request timed out. Please try again.[/red]")
        return None
    except requests.exceptions.RequestException as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Request interrupted by user[/yellow]")
        return None

def show_model_selection():
    console.print("\n[bold cyan]Available Models:[/bold cyan]")
    
    # Group models by local vs remote
    local_models = []
    remote_models = []
    
    for key, model in AVAILABLE_MODELS.items():
        if model["backend"] in ["ollama", "vllm"]:
            local_models.append((key, model))
        else:  # cloud/remote models
            remote_models.append((key, model))
    
    # Display local models
    console.print("\n[bold magenta]1. Locally Runned LLMs:[/bold magenta]")
    
    # Group local models by backend
    for backend in ["OLLAMA", "VLLM"]:
        backend_models = [(k, m) for k, m in local_models if m["backend"].upper() == backend]
        if backend_models:
            console.print(f"  [bold yellow]{backend}:[/bold yellow]")
            for key, model in backend_models:
                console.print(f"    {key}. {model['display']}")
    
    # Display remote models
    console.print("\n[bold magenta]2. Remote Models (supports Chain of Thought):[/bold magenta]")
    console.print("  [bold yellow]OPENROUTER:[/bold yellow]")
    for key, model in remote_models:
        console.print(f"    {key}. {model['display']}")
    
    try:
        choice = Prompt.ask("\nSelect model number", choices=list(AVAILABLE_MODELS.keys()))
        return AVAILABLE_MODELS[choice]
    except KeyboardInterrupt:
        console.print("\n[yellow]Model selection interrupted, exiting...[/yellow]")
        sys.exit(0)

def toggle_reasoning():
    try:
        choice = Prompt.ask(
            "\n[bold cyan]Enable step-by-step reasoning?[/bold cyan] [y/n]",
            choices=["y", "n"],
            default="n"
        )
        return choice == "y"
    except KeyboardInterrupt:
        console.print("\n[yellow]Reasoning selection interrupted, exiting...[/yellow]")
        sys.exit(0)

def main():
    try:
        # Check for API keys
        if not os.getenv('OPENROUTER_API_KEY'):
            console.print("[yellow]Note: OpenRouter models require OPENROUTER_API_KEY in .env file.[/yellow]")
        if not os.getenv('AVIAN_API_KEY'):
            console.print("[yellow]Note: Avian.ai models require AVIAN_API_KEY in .env file.[/yellow]")
        
        console.print("[bold blue]Welcome to Multi-Backend Chat Interface![/bold blue]")
        console.print("Commands:")
        console.print("  'quit' - Exit the chat")
        console.print("  'switch' - Switch between models")
        console.print("  'reason' - Toggle reasoning mode")
        console.print("  CTRL+C - Interrupt current operation or exit\n")
        
        current_model = show_model_selection()
        reasoning_enabled = toggle_reasoning()
        
        console.print(Panel(
            f"Using model: [green]{current_model['name']}[/green] with "
            f"{'provider' if current_model['backend'] == 'cloud' else 'backend'}: "
            f"[yellow]{current_model.get('provider', current_model['backend']).upper()}[/yellow]\n"
            f"Reasoning mode: [{'green' if reasoning_enabled else 'red'}]{'enabled' if reasoning_enabled else 'disabled'}[/{'green' if reasoning_enabled else 'red'}]"
        ))
        
        # Initialize vLLM instance if needed
        llm_instance = None
        if current_model["backend"] == "vllm":
            console.print("[yellow]Initializing vLLM (this might take a few moments)...[/yellow]")
            if is_apple_silicon():
                console.print("[yellow]Using CPU for vLLM on Apple Silicon (MPS not fully supported yet)[/yellow]")
                llm_instance = LLM(
                    model=current_model["name"],
                    device="cpu",
                    download_dir="models",
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    enforce_eager=True
                )
            else:
                llm_instance = LLM(model=current_model["name"])
            console.print("[green]vLLM initialized successfully![/green]")
        
        while True:
            try:
                user_input = Prompt.ask("[bold green]You")
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'switch':
                    current_model = show_model_selection()
                    reasoning_enabled = toggle_reasoning()
                    if current_model["backend"] == "vllm" and llm_instance is None:
                        console.print("[yellow]Initializing vLLM (this might take a few moments)...[/yellow]")
                        if is_apple_silicon():
                            console.print("[yellow]Using CPU for vLLM on Apple Silicon (MPS not fully supported yet)[/yellow]")
                            llm_instance = LLM(
                                model=current_model["name"],
                                device="cpu",
                                download_dir="models",
                                trust_remote_code=True,
                                tensor_parallel_size=1,
                                enforce_eager=True
                            )
                        else:
                            llm_instance = LLM(model=current_model["name"])
                        console.print("[green]vLLM initialized successfully![/green]")
                    console.print(Panel(
                        f"Switched to model: [green]{current_model['name']}[/green] with "
                        f"{'provider' if current_model['backend'] == 'cloud' else 'backend'}: "
                        f"[yellow]{current_model.get('provider', current_model['backend']).upper()}[/yellow]\n"
                        f"Reasoning mode: [{'green' if reasoning_enabled else 'red'}]{'enabled' if reasoning_enabled else 'disabled'}[/{'green' if reasoning_enabled else 'red'}]"
                    ))
                    continue
                
                if user_input.lower() == 'reason':
                    reasoning_enabled = toggle_reasoning()
                    console.print(Panel(
                        f"Reasoning mode: [{'green' if reasoning_enabled else 'red'}]{'enabled' if reasoning_enabled else 'disabled'}[/{'green' if reasoning_enabled else 'red'}]"
                    ))
                    continue
                    
                console.print("\n[bold yellow]Assistant: [/bold yellow]", end="")
                
                if current_model["backend"] == "ollama":
                    response = chat_with_ollama(user_input, current_model["name"])
                elif current_model["backend"] == "cloud":
                    if current_model["provider"] == "openrouter":
                        response = chat_with_openrouter(
                            user_input, 
                            current_model["name"],
                            current_model.get("provider_name"),
                            REASONING_PROMPT if reasoning_enabled else None
                        )
                    else:  # avian direct API (legacy)
                        response = chat_with_avian(user_input, current_model["name"])
                else:  # vllm
                    response = chat_with_vllm(user_input, current_model["name"], llm_instance)
                    
                if response:
                    console.print(Markdown(response))
                
                console.print()  # Empty line for better readability
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Current operation interrupted[/yellow]")
                continue
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    main() 