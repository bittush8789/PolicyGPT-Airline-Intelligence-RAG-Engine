"""
Lightweight document loader for S3-compatible storage.
Replaces unstructured library with lighter alternatives:
- PDF: pypdf
- HTML: BeautifulSoup4
- TXT: Direct reading
"""

import io
import logging
from typing import List, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class S3DocumentLoader:
    """Lightweight document loader for S3-compatible storage."""
    
    def __init__(self):
        """Initialize the S3 client."""
        s3_config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
        self.s3_client = boto3.client(
            's3',
            endpoint_url=settings.linode_object_storage_endpoint,
            aws_access_key_id=settings.linode_object_storage_access_key,
            aws_secret_access_key=settings.linode_object_storage_secret_key,
            config=s3_config
        )
        self.bucket = settings.linode_object_storage_bucket
    
    def load(self, key: str) -> List[Document]:
        """
        Load a document from S3 and parse it based on file extension.
        
        Args:
            key: S3 object key (file path)
            
        Returns:
            List of Document objects (usually one, but can be multiple for PDFs with pages)
        """
        try:
            # Download file from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            file_content = response['Body'].read()
            
            # Determine file type from extension
            file_ext = key.lower().split('.')[-1] if '.' in key else ''
            
            # Parse based on file type
            if file_ext == 'pdf':
                return self._parse_pdf(file_content, key)
            elif file_ext in ['html', 'htm']:
                return self._parse_html(file_content, key)
            elif file_ext == 'txt':
                return self._parse_text(file_content, key)
            else:
                # Try to parse as text for unknown types
                logger.warning(f"Unknown file type for {key}, attempting to parse as text")
                return self._parse_text(file_content, key)
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Failed to load {key} from S3: {error_code} - {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse {key}: {e}", exc_info=True)
            raise
    
    def _parse_pdf(self, content: bytes, source: str) -> List[Document]:
        """Parse PDF content using pypdf."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            
            documents = []
            full_text = []
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} of {source}: {e}")
                    continue
            
            # Combine all pages into a single document
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
                # Return empty document to avoid breaking the pipeline
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
            # Decode bytes to string
            html_content = content.decode('utf-8', errors='ignore')
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up multiple newlines
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
            # Decode bytes to string
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

