import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths based on your project structure
INPUT_FOLDER = "monolingual_texts/input_texts"
OUTPUT_FOLDER = "monolingual_texts/output_texts"

# Character definitions for replacement
# We replace standard punctuation with specific Nenets modifier letters
CHAR_APOSTROPHE_OLD = "'"       # Standard keyboard apostrophe
CHAR_APOSTROPHE_NEW = "\u02BC"  # Unicode: MODIFIER LETTER APOSTROPHE

CHAR_QUOTE_OLD = '"'            # Standard keyboard double quote
CHAR_QUOTE_NEW = "\u02EE"       # Unicode: MODIFIER LETTER DOUBLE APOSTROPHE

def normalize_text(text):
    """
    Replaces standard punctuation marks with Nenets-specific modifier letters.
    """
    # Replace single apostrophe
    text = text.replace(CHAR_APOSTROPHE_OLD, CHAR_APOSTROPHE_NEW)
    
    # Replace double quote
    text = text.replace(CHAR_QUOTE_OLD, CHAR_QUOTE_NEW)
    
    return text

def main():
    # 1. Create output directory if it does not exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"[INFO] Created output directory: {OUTPUT_FOLDER}")

    # 2. Validate input directory
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input directory '{INPUT_FOLDER}' not found.")
        return

    print(f"[INFO] Processing files in '{INPUT_FOLDER}'...\n")

    processed_count = 0
    ignored_count = 0
    
    # 3. Iterate through files in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        
        # FILTER: Ignore WSL/Windows metadata files (Zone.Identifier)
        if "Zone.Identifier" in filename:
            ignored_count += 1
            continue

        # FILTER: Process only .txt files
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            try:
                # Read original content
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply normalization
                new_content = normalize_text(content)
                
                # Write converted content to output folder
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"[OK] Processed: {filename}")
                processed_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")

    # 4. Final Summary
    print("-" * 30)
    print("PROCESSING COMPLETE")
    print(f"Files successfully converted: {processed_count}")
    print(f"System files ignored:         {ignored_count}")
    print(f"Output location:              {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()