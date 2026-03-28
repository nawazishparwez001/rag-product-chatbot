import requests
import yaml

# The GitHub API endpoint for this repo's contents.
# We use the API (not scraping the webpage) because it gives us a clean
# list of files in JSON format — much easier than parsing HTML.
REPO_API_BASE = "https://api.github.com/repos/LennysNewsletter/lennys-newsletterpodcastdata/contents"

# Raw content base URL — this is how you get the actual file text, not the GitHub UI page.
RAW_BASE = "https://raw.githubusercontent.com/LennysNewsletter/lennys-newsletterpodcastdata/main"

# The two folders we want to load from
FOLDERS = ["newsletters", "podcasts"]


def fetch_file_list(folder):
    """
    Ask the GitHub API: what .md files exist in this folder?
    Returns a list of raw file URLs.
    """
    url = f"{REPO_API_BASE}/{folder}"
    response = requests.get(url)

    # If GitHub returns an error (e.g. rate limit, wrong URL), stop early
    response.raise_for_status()

    files = response.json()

    # We only want .md files — skip anything else (like .json or .txt)
    return [
        f"{RAW_BASE}/{folder}/{f['name']}"
        for f in files
        if f["name"].endswith(".md")
    ]


def parse_markdown(raw_text):
    """
    Split a markdown file into two parts:
      - frontmatter: the YAML block between the --- markers (title, date, type, etc.)
      - body: the actual content below the frontmatter

    We separate these because metadata and content serve different purposes:
    metadata helps us filter/display results, content is what gets embedded and searched.
    """
    # Check if the file starts with a YAML frontmatter block
    if not raw_text.startswith("---"):
        return {}, raw_text.strip()

    # Find the closing --- to extract just the YAML portion
    end = raw_text.find("---", 3)
    if end == -1:
        return {}, raw_text.strip()

    yaml_block = raw_text[3:end].strip()
    body = raw_text[end + 3:].strip()

    # Parse the YAML string into a Python dictionary
    try:
        metadata = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        metadata = {}

    return metadata, body


def load_documents():
    """
    Main function: fetch all .md files from GitHub, parse each one,
    and return a list of document dicts.

    Each document looks like:
    {
        "title": "...",
        "type": "newsletter" or "podcast",
        "date": "...",
        "source_url": "...",
        "content": "the full text of the article/transcript"
    }
    """
    documents = []

    for folder in FOLDERS:
        print(f"Fetching file list from: {folder}/")
        file_urls = fetch_file_list(folder)

        for url in file_urls:
            print(f"  Loading: {url.split('/')[-1]}")
            response = requests.get(url)
            response.raise_for_status()

            metadata, body = parse_markdown(response.text)

            # Build a clean document — only keep fields we care about
            doc = {
                "title": metadata.get("title", "Untitled"),
                "type": metadata.get("type", folder.rstrip("s")),  # fallback: folder name
                "date": str(metadata.get("date", "")),
                "source_url": url,
                "content": body,
            }

            documents.append(doc)

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# This block only runs when you execute this file directly (not when imported).
# It's a quick way to test that the loader works before wiring it to the rest of the pipeline.
if __name__ == "__main__":
    docs = load_documents()

    # Print a preview of the first document to verify it loaded correctly
    if docs:
        print("\n--- Sample Document ---")
        print(f"Title : {docs[0]['title']}")
        print(f"Type  : {docs[0]['type']}")
        print(f"Date  : {docs[0]['date']}")
        print(f"URL   : {docs[0]['source_url']}")
        print(f"Body  : {docs[0]['content'][:300]}...")
