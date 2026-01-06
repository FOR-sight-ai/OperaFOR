import os
import json
import zipfile
import shutil
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import lxml.etree as ET

class DOCXHandler:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_xml_structure(self) -> Dict[str, Any]:
        """
        Extracts the XML structure of the DOCX, focusing on word/document.xml.
        Returns a dict with beautified XML content.
        """
        try:
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                # Primary interesting file is word/document.xml
                if 'word/document.xml' not in zip_ref.namelist():
                    return {"error": "word/document.xml not found in DOCX"}
                
                xml_content = zip_ref.read('word/document.xml')
                # Parse and beautify
                parser = ET.XMLParser(remove_blank_text=True)
                root = ET.fromstring(xml_content, parser)
                beautified_xml = ET.tostring(root, encoding='unicode', pretty_print=True)
                
                inventory = {
                    "filename": self.file_path.name,
                    "structure": {
                        "word/document.xml": beautified_xml
                    }
                }

                # Optionally add other parts if needed (styles, footers, etc.)
                # For now, focus on the main content
                
                return inventory
        except Exception as e:
            return {"error": f"Could not read DOCX XML: {e}"}

    def replace_text(self, replacements: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Replaces text in-place by manipulating the underlying XML files.
        'replacements' is a list of {"old_text": "...", "new_text": "..."}
        """
        if not replacements:
            return {"success": True, "message": "No replacements provided"}

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Unzip to temp dir
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # We'll touch word/document.xml, and maybe others like headers/footers
            files_to_check = list(temp_dir.glob('word/*.xml'))
            total_matches = 0

            for xml_file in files_to_check:
                content = xml_file.read_text(encoding='utf-8')
                new_content = content
                for rep in replacements:
                    old = rep.get("old_text")
                    new = rep.get("new_text")
                    if old and old in new_content:
                        # Simple string replacement in XML can be tricky due to tag splitting
                        # but if the user provides the "old_text" as seen in the XML dump,
                        # it should work or fail gracefully.
                        # Advanced logic would involve traversing the ET tree.
                        # For now, let's try direct string replacement first as it's more robust to XML nuances if the string is intact.
                        new_content = new_content.replace(old, new)
                        total_matches += 1
                
                if new_content != content:
                    xml_file.write_text(new_content, encoding='utf-8')

            if total_matches > 0:
                # Re-zip
                self._create_zip(temp_dir, self.file_path)
                return {"success": True, "message": f"Applied {total_matches} replacements across XML parts."}
            else:
                return {"success": True, "message": "No matches found in XML."}

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            shutil.rmtree(temp_dir)

    def _create_zip(self, source_dir: Path, output_file: Path):
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

def get_docx_xml(file_path: str) -> str:
    try:
        handler = DOCXHandler(file_path)
        structure = handler.get_xml_structure()
        return json.dumps(structure, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def edit_docx_xml(file_path: str, replacements: List[Dict[str, str]]) -> str:
    try:
        handler = DOCXHandler(file_path)
        result = handler.replace_text(replacements)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
