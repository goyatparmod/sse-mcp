import asyncio
import json
import os
import sys
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize AWS Bedrock client for Claude
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
    
        if not aws_access_key or not aws_secret_key:
            print("Warning: AWS credentials not found in environment variables.")
            print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        
        try:
            # Create Bedrock client with session token if provided
            client_kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': aws_region,
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key,
            }
            
            if aws_session_token:
                client_kwargs['aws_session_token'] = aws_session_token
                
            self.bedrock_client = boto3.client(**client_kwargs)
            
            # Test the connection to detect authentication issues early
            # self.bedrock_client.list_foundation_models(maxResults=1)
            # print("Successfully authenticated with AWS Bedrock")
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            print(f"AWS Bedrock authentication error ({error_code}): {error_message}")
            print("\nPossible solutions:")
            print("1. Check that your AWS credentials are correct and not expired")
            print("2. Make sure the IAM role/user has access to Bedrock")
            print("3. If using temporary credentials, ensure AWS_SESSION_TOKEN is set correctly")
            print("4. Verify you have access to the Claude 3.7 Sonnet model in your AWS region")
            
            # Still create the client, but we'll check bedrock availability before each call
            self.bedrock_client = boto3.client(**client_kwargs)
        
        self.claude_model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.bedrock_available = True  # Will be set to False if authentication fails during usage

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        try:
            print(f"Connecting to server at {server_url}...")
            # Store the context managers so they stay alive
            self._streams_context = sse_client(url=server_url)
            streams = await self._streams_context.__aenter__()

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()

            # Initialize
            await self.session.initialize()

            # List available tools to verify connection
            print("Initialized SSE client...")
            print("Listing tools...")
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            raise

    async def cleanup(self):
        """Properly clean up the session and streams"""
        try:
            if hasattr(self, '_session_context') and self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, '_streams_context') and self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def _invoke_bedrock_claude(self, messages, tools=None):
        """Invoke Claude 3.7 Sonnet via AWS Bedrock"""
        if not self.bedrock_available:
            raise Exception("AWS Bedrock authentication failed. Please check your credentials.")
            
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": messages
        }
        
        if tools:
            request_body["tools"] = tools
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.claude_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read().decode('utf-8'))
            return response_body
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code in ['UnrecognizedClientException', 'InvalidSignatureException', 
                              'ExpiredTokenException', 'AccessDeniedException']:
                self.bedrock_available = False
                print(f"AWS authentication error: {error_message}")
                print("Please check your AWS credentials and permissions.")
            
            raise Exception(f"AWS Bedrock error ({error_code}): {error_message}")

    def _serialize_for_json(self, obj: Any) -> Any:
        """Helper method to make objects JSON serializable"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        return str(obj)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        if not self.session:
            return "Not connected to server. Please connect first."
            
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        try:
            # Get available tools
            response = await self.session.list_tools()
            available_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]

            # Initial Claude API call via Bedrock
            print("Sending request to Claude via AWS Bedrock...")
            bedrock_response = self._invoke_bedrock_claude(
                messages=messages,
                tools=available_tools
            )
            
            # Process response and handle tool calls
            tool_results = []
            final_text = []

            for content in bedrock_response.get("content", []):
                
                if content.get("type") == 'text':
                    final_text.append(content.get("text", ""))
                elif content.get("type") == 'tool_use':
                    tool_name = content.get("name")
                    tool_args = content.get("input", {})
                    tool_id = content.get("id", f"tool_{len(tool_results)}")
                    
                    # Execute tool call
                    print(f"Calling tool: {tool_name}")
                    result = await self.session.call_tool(tool_name, tool_args)
                    
                    # Get the result content as a string to ensure it's serializable
                    result_content = str(result.content)
                    
                    # Display the tool call in the output
                    safe_args = json.dumps(tool_args, ensure_ascii=False)
                    final_text.append(f"[Calling tool {tool_name} with args {safe_args}]")

                    # Add the tool result to our conversation - include the required "id" field
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use", 
                                "name": tool_name, 
                                "input": tool_args,
                                "id": tool_id
                            }
                        ]
                    })
                    
                    # Pass the tool result as a string to ensure it's serializable - include the required "tool_use_id" field
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result", 
                                "content": result_content,
                                "tool_use_id": tool_id
                            }
                        ]
                    })

                    # Get next response from Claude via Bedrock
                    print("Getting Claude's response to the tool result...")
                    bedrock_response = self._invoke_bedrock_claude(messages=messages, tools=available_tools)
                    
                    # Add the final response to our output
                    for cont in bedrock_response.get("content", []):
                        if cont.get("type") == "text":
                            final_text.append(cont.get("text", ""))

            return "\n".join(final_text)
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            return f"Error processing query: {str(e)}\n\n{trace}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ('quit', 'exit'):
                    break
                    
                print("Processing your query...")
                response = await self.process_query(query)
                print("\n" + response)
                    
            except KeyboardInterrupt:
                print("\nDetected Ctrl+C. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server>")
        print("Example: uv run client.py http://localhost:8080/sse")
        print("\nPlease ensure you have set the following environment variables:")
        print("- AWS_ACCESS_KEY_ID")
        print("- AWS_SECRET_ACCESS_KEY")
        print("- AWS_REGION (optional, defaults to us-east-1)")
        sys.exit(1)

    server_url = sys.argv[1]
    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=server_url)
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\nDetected Ctrl+C. Shutting down gracefully...")
    finally:
        print("Cleaning up resources...")
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
