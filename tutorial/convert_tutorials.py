#!/usr/bin/env python3
"""
Convert egglog tutorial files (.egg) to HTML format.
Comments (lines starting with ;;) become text, code becomes markdown code blocks.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import markdown

def parse_egg_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse an .egg file and return a list of (type, content) tuples.
    type is either 'comment' or 'code'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    blocks = []
    current_block = []
    current_type = None
    
    for line in lines:
        stripped = line.strip()
        
        # Determine line type
        if stripped.startswith(';;'):
            line_type = 'comment'
            # Remove the comment prefix and clean up
            content = line[2:]
        elif stripped == '' and current_type == 'comment':
            # Empty line continues comment block
            line_type = 'comment'
            content = line
        elif stripped == '':
            # Empty line in code or between blocks
            if current_block:
                line_type = current_type
                content = line
            else:
                continue
        else:
            line_type = 'code'
            content = line
        
        # Handle block transitions
        if current_type is None:
            current_type = line_type
        elif current_type != line_type:
            # Type changed, save current block and start new one
            if current_block:
                blocks.append((current_type, ''.join(current_block)))
            current_block = []
            current_type = line_type
        
        current_block.append(content)
    
    # Add final block
    if current_block:
        blocks.append((current_type, ''.join(current_block)))

    return blocks

def extract_title(blocks: List[Tuple[str, str]]) -> str:
    """Extract title from the first comment block"""
    for block_type, content in blocks:
        if block_type == 'comment':
            # Get first non-empty line as title
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    return line
    return "Egglog Tutorial"

def blocks_to_html(blocks: List[Tuple[str, str]]) -> str:
    """Convert parsed blocks to HTML using markdown parser"""
    html_parts = []
    
    # Initialize markdown parser with useful extensions
    md = markdown.Markdown(extensions=['codehilite', 'fenced_code', 'tables'])
    
    for block_type, content in blocks:
        if block_type == 'comment':
            # Convert comment to markdown, then to HTML
            # content = content.strip()
            if content:
                print("=== Comment Block ===")
                print(content)
                print("======================")
                # Convert the comment content as markdown
                html_content = md.convert(content)
                html_parts.append(html_content)
        
        elif block_type == 'code':
            # Convert code to markdown code block
            content = content.rstrip()
            if content:
                # Use fenced code block syntax for better formatting
                markdown_code = f"```lisp\n{content}\n```"
                html_content = md.convert(markdown_code)
                html_parts.append(html_content)
    
    return '\n'.join(html_parts)

def get_tutorial_files(tutorial_dir: Path) -> List[Path]:
    """Get all .egg files in tutorial directory, sorted"""
    egg_files = list(tutorial_dir.glob('*.egg'))
    return sorted(egg_files)

def generate_navigation(current_file: Path, all_files: List[Path]) -> str:
    """Generate navigation bar"""
    nav_items = []
    
    for i, file_path in enumerate(all_files):
        name = file_path.stem
        title = name.replace('-', ' ').title()
        
        if file_path == current_file:
            nav_items.append(f'<span class="current">{title}</span>')
        else:
            html_name = f"{name}.html"
            nav_items.append(f'<a href="{html_name}">{title}</a>')
    
    return f'''
    <nav class="tutorial-nav">
        <div class="nav-container">
            <h1>Egglog Tutorials</h1>
            <div class="nav-links">
                {' | '.join(nav_items)}
            </div>
        </div>
    </nav>
    '''

def generate_html_page(file_path: Path, all_files: List[Path]) -> str:
    """Generate complete HTML page for a tutorial file"""
    blocks = parse_egg_file(file_path)
    title = extract_title(blocks)
    content = blocks_to_html(blocks)
    navigation = generate_navigation(file_path, all_files)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Egglog Tutorial</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 0 20px;
            background-color: #fafafa;
        }}
        
        .tutorial-nav {{
            background: #2c3e50;
            color: white;
            padding: 1rem 0;
            margin: -20px -20px 2rem -20px;
            border-bottom: 3px solid #3498db;
        }}
        
        .nav-container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .tutorial-nav h1 {{
            margin: 0 0 0.5rem 0;
            color: #ecf0f1;
            font-size: 1.5rem;
        }}
        
        .nav-links a {{
            color: #3498db;
            text-decoration: none;
            margin: 0 0.5rem;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        
        .nav-links a:hover {{
            background-color: rgba(52, 152, 219, 0.2);
        }}
        
        .nav-links .current {{
            background-color: #3498db;
            color: white;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            margin: 0 0.5rem;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }}
        
        p {{
            margin: 1rem 0;
            text-align: justify;
        }}
        
        code {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            padding: 0.2rem 0.4rem;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
        }}
        
        pre {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}
        
        pre code {{
            background: none;
            border: none;
            padding: 0;
            font-size: 0.9rem;
            line-height: 1.4;
        }}
        
        .language-lisp {{
            color: #2c3e50;
        }}
        
        footer {{
            margin-top: 3rem;
            padding: 2rem 0;
            text-align: center;
            border-top: 1px solid #e9ecef;
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 600px) {{
            .nav-links {{
                font-size: 0.9rem;
            }}
            
            .nav-links a, .nav-links .current {{
                margin: 0 0.2rem;
                padding: 0.2rem 0.4rem;
            }}
        }}
    </style>
</head>
<body>
    {navigation}
    
    <main>
        <h1>{title}</h1>
        {content}
    </main>
    
    <footer>
        <p>Generated from <code>{file_path.name}</code> | <a href="https://github.com/egraphs-good/egglog" target="_blank">Egglog Project</a></p>
    </footer>
</body>
</html>'''

def main():
    """Main conversion function"""
    # Get tutorial directory
    tutorial_dir = Path(__file__).parent
    
    if not tutorial_dir.exists():
        print(f"Tutorial directory not found: {tutorial_dir}")
        return
    
    # Get all tutorial files
    tutorial_files = get_tutorial_files(tutorial_dir)
    
    if not tutorial_files:
        print("No .egg files found in tutorial directory")
        return
    
    print(f"Found {len(tutorial_files)} tutorial files:")
    for file_path in tutorial_files:
        print(f"  - {file_path.name}")
    
    # Convert each file
    for file_path in tutorial_files:
        html_content = generate_html_page(file_path, tutorial_files)
        
        # Write HTML file
        output_file = file_path.with_suffix('.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Generated: {output_file}")
    
    print("\nConversion complete! Open any .html file in your browser.")

if __name__ == '__main__':
    main()