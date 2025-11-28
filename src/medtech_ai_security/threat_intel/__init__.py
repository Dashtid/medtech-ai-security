"""
Threat Intelligence Module

NLP-based extraction and classification of medical device vulnerabilities
from CVE databases, ICS-CERT advisories, and manufacturer bulletins.
"""

from medtech_ai_security.threat_intel.nvd_scraper import NVDScraper
from medtech_ai_security.threat_intel.cisa_scraper import CISAScraper

__all__ = ["NVDScraper", "CISAScraper"]
