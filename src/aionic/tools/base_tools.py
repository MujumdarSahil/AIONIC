"""
Base Tool Implementations - Common tools for agent use.

Provides standard tools that agents can use for common operations.
"""

from typing import Any, Dict
import json
import time

from ..core.tool import ToolInterface, ToolResult, ToolCategory


class WebSearchTool(ToolInterface):
    """Tool for performing web searches."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information on a given query"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.RESEARCH
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5,
                "required": False,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "LOW"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute web search (placeholder implementation)."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        # Placeholder: In production, integrate with search API
        # For now, return mock results
        start_time = time.time()
        
        mock_results = [
            {"title": f"Result {i} for {query}", "url": f"https://example.com/{i}"}
            for i in range(min(max_results, 5))
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            data={"results": mock_results, "query": query},
            execution_time_ms=execution_time,
        )


class FileReadTool(ToolInterface):
    """Tool for reading files."""
    
    @property
    def name(self) -> str:
        return "file_read"
    
    @property
    def description(self) -> str:
        return "Read contents of a file"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.INFORMATION
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to file to read",
                "required": True,
            },
            "encoding": {
                "type": "string",
                "description": "File encoding",
                "default": "utf-8",
                "required": False,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "MEDIUM"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute file read (placeholder - should validate paths)."""
        file_path = kwargs.get("file_path", "")
        encoding = kwargs.get("encoding", "utf-8")
        
        start_time = time.time()
        
        try:
            # Placeholder: In production, add path validation and security checks
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={"content": content, "file_path": file_path},
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )


class DataAnalysisTool(ToolInterface):
    """Tool for basic data analysis operations."""
    
    @property
    def name(self) -> str:
        return "data_analysis"
    
    @property
    def description(self) -> str:
        return "Perform statistical analysis on data"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "data": {
                "type": "array",
                "description": "Data to analyze",
                "required": True,
            },
            "operation": {
                "type": "string",
                "description": "Analysis operation (mean, median, std, etc.)",
                "required": True,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "LOW"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute data analysis."""
        data = kwargs.get("data", [])
        operation = kwargs.get("operation", "mean")
        
        start_time = time.time()
        
        try:
            if not isinstance(data, list):
                raise ValueError("Data must be a list")
            
            numeric_data = [float(x) for x in data if isinstance(x, (int, float, str))]
            
            if not numeric_data:
                raise ValueError("No numeric data found")
            
            result_value = None
            if operation == "mean":
                result_value = sum(numeric_data) / len(numeric_data)
            elif operation == "median":
                sorted_data = sorted(numeric_data)
                n = len(sorted_data)
                result_value = (sorted_data[n//2] + sorted_data[(n-1)//2]) / 2
            elif operation == "std":
                mean = sum(numeric_data) / len(numeric_data)
                variance = sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)
                result_value = variance ** 0.5
            elif operation == "min":
                result_value = min(numeric_data)
            elif operation == "max":
                result_value = max(numeric_data)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={"operation": operation, "result": result_value, "data_points": len(numeric_data)},
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )


class CodeExecutionTool(ToolInterface):
    """Tool for executing code (high risk)."""
    
    @property
    def name(self) -> str:
        return "code_execution"
    
    @property
    def description(self) -> str:
        return "Execute Python code (sandboxed)"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.COMPUTATION
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 10,
                "required": False,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "CRITICAL"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute code (placeholder - should use sandbox in production)."""
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 10)
        
        start_time = time.time()
        
        # WARNING: This is a placeholder. In production, use proper sandboxing
        try:
            # Placeholder implementation - DO NOT use exec() in production
            # Should use restricted execution environment
            namespace = {}
            exec(code, namespace)
            result = namespace.get("result", "Code executed successfully")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={"result": str(result)},
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )


class DatabaseQueryTool(ToolInterface):
    """Tool for querying databases."""
    
    @property
    def name(self) -> str:
        return "database_query"
    
    @property
    def description(self) -> str:
        return "Execute SQL query on database"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.INFORMATION
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "SQL query to execute",
                "required": True,
            },
            "database": {
                "type": "string",
                "description": "Database name",
                "required": False,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "HIGH"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute database query (placeholder)."""
        query = kwargs.get("query", "")
        database = kwargs.get("database", "default")
        
        start_time = time.time()
        
        # Placeholder: In production, use actual database connection with proper security
        # Mock result
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            data={"query": query, "database": database, "rows": 0},
            metadata={"warning": "Placeholder implementation"},
            execution_time_ms=execution_time,
        )

