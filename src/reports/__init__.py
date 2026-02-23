"""Reports package for Amazon Sales Data Analysis.

This package handles report generation and summaries.
"""

from .summary import ReportGenerator, generate_summary

__all__ = [
    "ReportGenerator",
    "generate_summary",
]
