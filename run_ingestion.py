"""
run_ingestion.py

Interactive script to run the Document Ingestion Pipeline.
Prompts the user to select or input an insurance policy PDF, processes it,
and displays the extracted metadata.
"""

import os
import sys

# Ensure the workspace directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion import process_policy_pdf

def select_file_gui() -> str:
    """Attempts to open a GUI file dialog to select a PDF. Falls back to terminal input on failure."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Initialize root window and hide it immediately
        root = tk.Tk()
        root.withdraw()
        # Force the dialog to appear on top of other windows
        root.attributes("-topmost", True)
        
        print("Opening file selection dialog (GUI)...")
        file_path = filedialog.askopenfilename(
            title="Select Policy PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        return file_path
    except Exception:
        # Fallback to CLI input if Tkinter / GUI is not supported
        return ""

def main():
    print("==================================================")
    print("   Insurance Policy Document Ingestion Utility    ")
    print("==================================================")
    
    # 1. Request file path (Try GUI first, fallback to CLI)
    pdf_path = select_file_gui()
    
    if not pdf_path:
        print("GUI dialog closed or unavailable. Please enter the file path manually.")
        pdf_path = input("Enter the path to the policy PDF: ").strip()
        # Clean quotes if dragged and dropped
        pdf_path = pdf_path.strip("'\"")
        
    if not pdf_path:
        print("Error: No file selected. Exiting.")
        sys.exit(1)
        
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'. Exiting.")
        sys.exit(1)
        
    print(f"\nProcessing policy PDF: {os.path.basename(pdf_path)}...")
    
    try:
        # 2. Run the ingestion pipeline
        text, metadata = process_policy_pdf(pdf_path)
        
        # 3. Print the results beautifully
        print("\n==================================================")
        print("         Extracted Metadata (Saved to Catalog)      ")
        print("==================================================")
        print(f"Policy ID:           {metadata.policy_id}")
        print(f"Policy Name:         {metadata.policy_name}")
        print(f"Insurer:             {metadata.insurer}")
        print(f"Plan Type:           {metadata.plan_type}")
        print(f"Premium:             ₹{metadata.premium:,.2f}")
        print(f"Min / Max Age:       {metadata.min_age} - {metadata.max_age} years")
        print(f"Sum Insured:         ₹{metadata.sum_insured:,.2f}")
        print(f"Smoker Allowed:      {'Yes' if metadata.smoker_allowed else 'No'}")
        print(f"Covers Diabetes:     {'Yes' if metadata.covers_diabetes else 'No'}")
        print(f"Covers Hypertension: {'Yes' if metadata.covers_hypertension else 'No'}")
        print(f"Parents Allowed:     {'Yes' if metadata.parents_allowed else 'No'}")
        print(f"Children Allowed:    {'Yes' if metadata.children_allowed else 'No'}")
        print("==================================================")
        print("Successfully processed and updated policy catalog, ChromaDB & PostgreSQL document store.")
        print(f"Total extracted text length: {len(text)} characters.")

        
    except Exception as e:
        print(f"\nError processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
