"""
Document loader for local documents directory.
Parses files based on extension:
- PDF: pypdf
- HTML: BeautifulSoup4
- TXT: Direct reading
"""

import io
import logging
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentLoader:
    """Document loader for local filesystem documents."""
    
    def __init__(self, documents_dir: Optional[Union[str, Path]] = None):
        """Initialize the document loader."""
        if documents_dir is None:
            self.documents_dir = Path(settings.documents_dir)
        else:
            self.documents_dir = Path(documents_dir)
    
    def load(self, file_path_or_name: Union[str, Path]) -> List[Document]:
        """
        Load a document from the local filesystem and parse it based on file extension.
        
        Args:
            file_path_or_name: Relative filename or absolute path to the document
            
        Returns:
            List of Document objects
        """
        path = Path(file_path_or_name)
        if not path.is_absolute():
            path = self.documents_dir / path
            
        filename = path.name
        
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")
            
        try:
            with open(path, "rb") as f:
                file_content = f.read()
                
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            
            if file_ext == 'pdf':
                return self._parse_pdf(file_content, filename)
            elif file_ext in ['html', 'htm']:
                return self._parse_html(file_content, filename)
            elif file_ext == 'txt':
                return self._parse_text(file_content, filename)
            else:
                logger.warning(f"Unknown file type for {filename}, attempting to parse as text")
                return self._parse_text(file_content, filename)
                
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}", exc_info=True)
            raise
    
    def _parse_pdf(self, content: bytes, source: str) -> List[Document]:
        """Parse PDF content using pypdf."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            
            documents = []
            full_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} of {source}: {e}")
                    continue
            
            if full_text:
                combined_text = "\n\n".join(full_text)
                documents.append(Document(
                    page_content=combined_text,
                    metadata={
                        "source": source,
                        "file_type": "pdf",
                        "page_count": len(pdf_reader.pages)
                    }
                ))
            else:
                logger.warning(f"No text extracted from PDF: {source}")
                documents.append(Document(
                    page_content="",
                    metadata={
                        "source": source,
                        "file_type": "pdf",
                        "page_count": 0
                    }
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {source}: {e}", exc_info=True)
            raise
    
    def _parse_html(self, content: bytes, source: str) -> List[Document]:
        """Parse HTML content using BeautifulSoup4."""
        try:
            html_content = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            if not cleaned_text:
                logger.warning(f"No text extracted from HTML: {source}")
                cleaned_text = ""
            
            return [Document(
                page_content=cleaned_text,
                metadata={
                    "source": source,
                    "file_type": "html"
                }
            )]
            
        except Exception as e:
            logger.error(f"Failed to parse HTML {source}: {e}", exc_info=True)
            raise
    
    def _parse_text(self, content: bytes, source: str) -> List[Document]:
        """Parse plain text content."""
        try:
            text = content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                logger.warning(f"Empty text file: {source}")
                text = ""
            
            return [Document(
                page_content=text,
                metadata={
                    "source": source,
                    "file_type": "txt"
                }
            )]
            
        except Exception as e:
            logger.error(f"Failed to parse text {source}: {e}", exc_info=True)
            raise


# Backwards compatibility alias
S3DocumentLoader = DocumentLoader
