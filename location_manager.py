from typing import Any, Dict, Optional
import httpx
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastMCP server for Location tools (SSE)
mcp = FastMCP("location")

# Constants
LOCATION_API_BASE = "https://int.dev.api.coxautoinc.com/wholesale-marketplace/enablement/locations"
BEARER_TOKEN = os.getenv("BEARER_TOKEN")


async def make_location_request(url: str) -> Dict[str, Any] | None:
    """Make a request to the Location API with proper error handling."""
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Accept": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching location data: {e}")
            return None


def format_location(location_data: Dict[str, Any]) -> str:
    """Format location data into a readable string."""
    address = location_data.get('address', {})
    contact = location_data.get('contact', {})
    
    return f"""
Location Name: {location_data.get('locationName', 'Unknown')}
Location Code: {location_data.get('locationCode', 'Unknown')}
Type: {location_data.get('locationType', 'Unknown')}
SubType: {location_data.get('locationSubType', 'Unknown')}
Operated By: {location_data.get('operatedBy', 'Unknown')}

Address: {address.get('address1', 'Unknown')},
         {address.get('city', 'Unknown')},
         {address.get('stateProvinceRegion', 'Unknown')} {address.get('postalCode', 'Unknown')}
Country: {address.get('country', 'Unknown')}
Coordinates: {address.get('latitude', 'Unknown')}, {address.get('longitude', 'Unknown')}

Contact: 
  Phone: {contact.get('phone', location_data.get('contactPhone', 'Unknown'))}
  Fax: {contact.get('fax', 'Not provided')}
  Email: {contact.get('email', 'Not provided')}

Time Zone: {location_data.get('timeZone', 'Unknown')}
Location Active: {location_data.get('locationActive', 'Unknown')}
Region: {location_data.get('region', 'Unknown')}
"""


@mcp.tool()
async def get_location_by_id(location_id: str) -> str:
    """Get location information by location ID/code.

    Args:
        location_id: Location identifier (e.g. QIM4, ABC1)
    """
    url = f"{LOCATION_API_BASE}/id/{location_id}"
    data = await make_location_request(url)

    if not data:
        return "Unable to fetch location data for this ID."

    # return format_location(data)
    retrurn data


@mcp.tool()
async def search_locations_by_name(name: str) -> str:
    """Search for locations by name.

    Args:
        name: Location name to search for (e.g. Manheim)
    """
    url = f"{LOCATION_API_BASE}/search?name={name}"
    data = await make_location_request(url)

    if not data or "items" not in data:
        return "Unable to fetch locations or no locations found."

    if not data["items"]:
        return f"No locations found matching '{name}'."

    locations = [format_location(location) for location in data["items"]]
    return "\n---\n".join(locations)


@mcp.tool()
async def get_locations_by_state(state: str) -> str:
    """Get locations in a specific state.

    Args:
        state: Two-letter state code (e.g. MS, FL, CA)
    """
    url = f"{LOCATION_API_BASE}/search?state={state}"
    data = await make_location_request(url)

    if not data or "items" not in data:
        return f"Unable to fetch locations for state '{state}'."

    if not data["items"]:
        return f"No locations found in state '{state}'."

    locations = [format_location(location) for location in data["items"]]
    return "\n---\n".join(locations)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCP SSE-based Location server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)