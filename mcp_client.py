import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            print("✅ Connected to MCP Server")

            # Step 1: Run SEO Analysis
            print("🔍 Running SEO Analysis via MCP...")
            seo_result = await session.call_tool(
                "seo_analyzer",
                arguments={
                    "url": "https://amazon.com",
                    "primary_keyword": "SEO optimization",
                    "business_goals": {
                        "primary_objective": "increase_sales",
                        "target_audience": "b2b_smb",
                        "conversion_goal": "form_submission",
                        "content_strategy": "educational",
                        "business_stage": "growth",
                        "competitive_position": "challenger"
                    }
                }
            )
            print("✅ SEO Analysis Done")
            print(f"SEO Score: {seo_result.content[0].text if seo_result.content else 'No data'}")

            # Step 2: Export to Google Drive
            print("\n📤 Exporting to Google Drive...")
            
            # Parse the SEO result (it comes as TextContent)
            import json
            seo_data = json.loads(seo_result.content[0].text)
            
            export_result = await session.call_tool(
                "export_seo_to_google_drive",
                arguments={
                    "seo_data": seo_data,
                    "website_name": "amazon"
                }
            )
            print("✅ Export Done!")
            
            export_data = json.loads(export_result.content[0].text)
            if export_data.get('success'):
                print(f"✅ Successfully exported to: {export_data.get('filename')}")
                print(f"📁 Google Drive Link: {export_data.get('file_url')}")
            else:
                print(f"❌ Export failed: {export_data.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())