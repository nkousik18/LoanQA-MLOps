"""
API routes for complete document extraction
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import logging
import os
import sys
import json
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.complete_document_extractor import CompleteDocumentExtractor
from api.document_ingestion import DocumentUploadHandler, ValidationError
from api.batch_processor import BatchProcessingHandler
from storage.storage_service import StorageService
from normalization.comparison_service import ComparisonService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
extractor = CompleteDocumentExtractor()
upload_handler = DocumentUploadHandler()
storage_service = StorageService()
batch_processor = BatchProcessingHandler(upload_handler, storage_service)
comparison_service = ComparisonService()


# ========================================================================
# Document Upload Endpoints
# ========================================================================

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a single loan document.
    
    Validates format, stores document, and returns metadata.
    Supports PDF, JPEG, PNG, TIFF formats.
    
    Requirements: 2.1, 8.3
    """
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Validate and upload document
        metadata = upload_handler.upload_document(
            file_data=file_content,
            file_name=file.filename,
            content_type=file.content_type
        )
        
        # Store document in storage service
        document_id = storage_service.store_document(
            file_data=file_content,
            file_name=metadata.file_name,
            file_type=metadata.file_type,
            page_count=metadata.page_count,
            document_id=metadata.document_id
        )
        
        logger.info(f"Document uploaded successfully: {document_id}")
        
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "Document uploaded successfully",
                "document_id": document_id,
                "metadata": metadata.to_dict()
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """
    Complete document extraction with storage integration.
    
    Uploads document, extracts ALL text, data, numbers, tables, boxes, nested columns.
    Uses both Form Parser and Document OCR.
    Stores results and returns complete extraction with accuracy metrics.
    
    Requirements: 2.1, 8.3
    """
    try:
        logger.info(f"Starting complete extraction: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Step 1: Upload and validate document
        metadata = upload_handler.upload_document(
            file_data=file_content,
            file_name=file.filename,
            content_type=file.content_type
        )
        
        # Step 2: Store document
        document_id = storage_service.store_document(
            file_data=file_content,
            file_name=metadata.file_name,
            file_type=metadata.file_type,
            page_count=metadata.page_count,
            document_id=metadata.document_id
        )
        
        # Step 3: Update status to processing
        storage_service.update_document_status(document_id, 'processing')
        
        # Step 4: Extract complete document
        mime_type = file.content_type or "application/pdf"
        result = extractor.extract_complete_document(
            file_content,
            mime_type,
            file.filename
        )
        
        # Step 5: Update status to completed
        storage_service.update_document_status(document_id, 'completed')
        
        logger.info(f"Extraction complete: {file.filename}")
        logger.info(f"Accuracy: {result.get('accuracy_metrics', {}).get('overall_accuracy', 0):.2%}")
        
        # Add document ID to result
        result['document_id'] = document_id
        
        # Return complete extraction
        return JSONResponse(content=result)
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        if 'document_id' in locals():
            storage_service.update_document_status(document_id, 'failed')
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# Batch Processing Endpoints
@router.get("/accuracy")
async def get_accuracy_info():
    """Get information about accuracy calculation"""
    return {
        "accuracy_metrics": {
            "overall_accuracy": "Combined confidence from both processors",
            "form_parser_accuracy": "Form Parser confidence score",
            "ocr_accuracy": "Document OCR confidence score",
            "text_extraction_confidence": "Confidence for all text elements",
            "table_extraction_confidence": "Confidence for table data",
            "form_field_confidence": "Confidence for form fields",
            "page_confidences": "Per-page confidence scores",
            "low_confidence_items": "Items with confidence < 85%"
        },
        "processors": {
            "form_parser": "Best for structured forms and tables",
            "document_ocr": "Best for general text extraction"
        },
        "validation": "Both processors used for cross-validation"
    }


@router.post("/compare")
async def compare_loans(loan_data_list: List[Dict]):
    """
    Compare multiple loan documents
    
    Accepts a list of loan data dictionaries and returns comparison results
    including best options by cost and flexibility, and pros/cons for each loan.
    """
    try:
        if len(loan_data_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 loans required for comparison"
            )
        
        logger.info(f"Comparing {len(loan_data_list)} loans")
        
        # Calculate metrics for each loan
        metrics_list = []
        for loan_data in loan_data_list:
            metrics = calculate_loan_metrics(loan_data)
            metrics_list.append(metrics)
        
        # Find best options
        best_by_cost = min(metrics_list, key=lambda m: m["total_cost_estimate"])
        best_by_flexibility = max(metrics_list, key=lambda m: m["flexibility_score"])
        
        # Generate comparison result
        result = {
            "loans": loan_data_list,
            "metrics": metrics_list,
            "best_by_cost": best_by_cost.get("loan_id", ""),
            "best_by_flexibility": best_by_flexibility.get("loan_id", ""),
            "comparison_notes": {
                "summary": f"Compared {len(loan_data_list)} loan options",
                "cost_range": f"Total cost ranges from ${min(m['total_cost_estimate'] for m in metrics_list):,.2f} to ${max(m['total_cost_estimate'] for m in metrics_list):,.2f}",
                "recommendation": "Review the comparison table and pros/cons for each loan"
            }
        }
        
        logger.info(f"Comparison complete - Best by cost: {result['best_by_cost']}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loans")
async def query_loans(
    loan_type: str = None,
    bank_name: str = None,
    min_amount: float = None,
    max_amount: float = None,
    skip: int = 0,
    limit: int = 100
):
    """
    Query loans with optional filters
    
    Supports filtering by loan type, bank name, and amount range.
    Returns paginated results.
    """
    try:
        logger.info(f"Querying loans with filters: type={loan_type}, bank={bank_name}, amount={min_amount}-{max_amount}")
        
        # In a real implementation, this would query a database
        # For now, return a sample response structure
        result = {
            "loans": [],
            "total": 0,
            "skip": skip,
            "limit": limit,
            "filters": {
                "loan_type": loan_type,
                "bank_name": bank_name,
                "min_amount": min_amount,
                "max_amount": max_amount
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_loan_metrics(loan_data: Dict) -> Dict:
    """
    Calculate comparison metrics for a loan
    
    Args:
        loan_data: Loan data dictionary
        
    Returns:
        Dictionary with calculated metrics
    """
    principal = loan_data.get("principal_amount", 0)
    interest_rate = loan_data.get("interest_rate", 0)
    tenure_months = loan_data.get("tenure_months", 0)
    processing_fee = loan_data.get("processing_fee", 0)
    moratorium_months = loan_data.get("moratorium_period_months", 0)
    
    # Calculate monthly EMI
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate > 0 and tenure_months > 0:
        monthly_emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    else:
        monthly_emi = principal / tenure_months if tenure_months > 0 else 0
    
    # Calculate total amounts
    total_amount = monthly_emi * tenure_months
    total_interest = total_amount - principal
    total_cost = total_amount + processing_fee
    effective_rate = (total_interest / principal * 100) if principal > 0 else 0
    
    # Calculate flexibility score (0-10)
    flexibility_score = 0.0
    
    # Moratorium period (up to 3 points)
    if moratorium_months > 0:
        flexibility_score += min(moratorium_months / 12 * 3, 3)
    
    # Prepayment penalty (3 points if no penalty)
    prepayment_penalty = loan_data.get("prepayment_penalty", "")
    if prepayment_penalty and ("no penalty" in prepayment_penalty.lower() or "nil" in prepayment_penalty.lower()):
        flexibility_score += 3
    
    # No collateral (2 points)
    if not loan_data.get("collateral_details"):
        flexibility_score += 2
    
    # No co-signer (2 points)
    if not loan_data.get("co_signer"):
        flexibility_score += 2
    
    return {
        "loan_id": loan_data.get("loan_id", loan_data.get("document_id", "unknown")),
        "total_cost_estimate": total_cost,
        "effective_interest_rate": effective_rate,
        "flexibility_score": flexibility_score,
        "monthly_emi": monthly_emi,
        "total_interest_payable": total_interest
    }


@router.post("/batch-upload")
async def batch_upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents for batch processing."""
    try:
        documents = []
        for file in files:
            file_content = await file.read()
            documents.append({
                    'file_name': file.filename,
                    'file_data': file_content
                })
        
        # Process batch
        summary = batch_processor.process_batch_from_bytes(
            documents=documents,
            continue_on_failure=True
        )
        
        logger.info(
            f"Batch upload complete: {summary.successful_documents} successful, "
            f"{summary.failed_documents} failed"
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "message": "Batch processing completed",
                "summary": summary.to_dict()
            }
        )
        
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


# ========================================================================
# Document Retrieval Endpoints
# ========================================================================

@router.get("/documents/{document_id}")
async def get_document_metadata(document_id: str):
    """
    Retrieve document metadata by ID.
    
    Returns document information without file content.
    
    Requirements: 10.1
    """
    try:
        logger.info(f"Retrieving document metadata: {document_id}")
        
        metadata = storage_service.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        return JSONResponse(content=metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/extracted-data")
async def get_extracted_data(document_id: str):
    """
    Get extracted loan data for a document.
    
    Returns normalized loan data if extraction is complete.
    
    Requirements: 10.1
    """
    try:
        logger.info(f"Retrieving extracted data for document: {document_id}")
        
        # First check if document exists
        metadata = storage_service.get_document_metadata(document_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        # Query loans by document ID
        loans = storage_service.query_loans()
        document_loans = [loan for loan in loans if loan.get('document_id') == document_id]
        
        if not document_loans:
            raise HTTPException(
                status_code=404,
                detail=f"No extracted data found for document: {document_id}"
            )
        
        return JSONResponse(content=document_loans[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving extracted data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loans")
async def query_loans(
    loan_type: Optional[str] = Query(None, description="Filter by loan type"),
    bank_name: Optional[str] = Query(None, description="Filter by bank name"),
    min_principal: Optional[float] = Query(None, description="Minimum principal amount"),
    max_principal: Optional[float] = Query(None, description="Maximum principal amount"),
    limit: int = Query(100, description="Maximum number of results")
):
    """
    Query loans with filters.
    
    Supports filtering by loan type, bank name, and principal amount range.
    
    Requirements: 10.1
    """
    try:
        logger.info(f"Querying loans with filters")
        
        loans = storage_service.query_loans(
            loan_type=loan_type,
            bank_name=bank_name,
            min_principal=min_principal,
            max_principal=max_principal,
            limit=limit
        )
        
        return JSONResponse(
            content={
                "count": len(loans),
                "loans": loans
            }
        )
        
    except Exception as e:
        logger.error(f"Error querying loans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# Comparison Endpoint
# ========================================================================

@router.post("/compare")
async def compare_loans(loan_ids: List[str]):
    """
    Compare multiple loan documents.
    
    Generates side-by-side comparison with metrics, best options,
    and detailed pros/cons analysis.
    
    Requirements: 2.1
    """
    try:
        logger.info(f"Comparing {len(loan_ids)} loans")
        
        if len(loan_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 loan IDs required for comparison"
            )
        
        # Retrieve loan data for all IDs
        loan_data_list = []
        for loan_id in loan_ids:
            loan_data = storage_service.get_loan_data(loan_id)
            if not loan_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Loan not found: {loan_id}"
                )
            loan_data_list.append(loan_data)
        
        # Generate comparison table
        comparison_table = {
            "headers": ["Loan ID", "Bank", "Principal", "Interest Rate", "Tenure"],
            "rows": []
        }
        
        for loan in loan_data_list:
            row = [
                loan.get('loan_id', 'N/A'),
                loan.get('bank_name', 'N/A'),
                f"{loan.get('principal_amount', 0):,.2f}",
                f"{loan.get('interest_rate', 0):.2f}%",
                f"{loan.get('tenure_months', 0)} months"
            ]
            comparison_table['rows'].append(row)
        
        logger.info(f"Comparison complete for {len(loan_ids)} loans")
        
        return JSONResponse(
            content={
                "status": "success",
                "loan_count": len(loan_ids),
                "comparison_table": comparison_table,
                "loans": loan_data_list
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# Processing Status Endpoints
# ========================================================================

@router.get("/processing-status/{job_id}")
async def get_processing_status(job_id: str):
    """
    Check processing status for a batch job.
    
    Returns job status, progress, and results.
    
    Requirements: 10.1
    """
    try:
        logger.info(f"Checking processing status: {job_id}")
        
        job_status = storage_service.get_processing_job(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing job not found: {job_id}"
            )
        
        return JSONResponse(content=job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """
    Get processing status for a specific document.
    
    Returns current processing status (pending, processing, completed, failed).
    
    Requirements: 10.1
    """
    try:
        logger.info(f"Checking document status: {document_id}")
        
        metadata = storage_service.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        return JSONResponse(
            content={
                "document_id": document_id,
                "status": metadata.get('processing_status', 'unknown'),
                "file_name": metadata.get('file_name'),
                "upload_timestamp": metadata.get('upload_timestamp')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# Legacy Extraction Endpoints (for backward compatibility)
# ========================================================================

@router.post("/extract/formatted")
async def extract_document_formatted(file: UploadFile = File(...)):
    """
    Extract and return formatted JSON string.
    
    Legacy endpoint for backward compatibility.
    """
    try:
        logger.info(f"Formatted extraction: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Determine MIME type
        mime_type = file.content_type or "application/pdf"
        
        # Extract complete document
        result = extractor.extract_complete_document(
            file_content,
            mime_type,
            file.filename
        )
        
        # Format as JSON string
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        
        # Return formatted JSON
        return JSONResponse(
            content={"formatted_json": json_output},
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Formatted extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/save")
async def extract_and_save(file: UploadFile = File(...)):
    """
    Extract and save complete extraction as JSON file.
    
    Legacy endpoint for backward compatibility.
    """
    try:
        logger.info(f"Extract and save: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Determine MIME type
        mime_type = file.content_type or "application/pdf"
        
        # Extract complete document
        result = extractor.extract_complete_document(
            file_content,
            mime_type,
            file.filename
        )
        
        # Create output path
        output_filename = f"{os.path.splitext(file.filename)[0]}_complete_extraction.json"
        output_path = os.path.join("/app", "output", output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to: {output_path}")
        
        return {
            "status": "success",
            "message": "Complete extraction saved",
            "output_file": output_path,
            "accuracy": result.get("accuracy_metrics", {}).get("overall_accuracy", 0),
            "processors_used": result.get("processors_used", [])
        }
        
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# Health and Info Endpoints
# ========================================================================

@router.get("/health")
async def health_check():
    """
    Health check for the API service.
    
    Returns service status and available features.
    """
    try:
        # Get storage statistics
        stats = storage_service.get_stats()
        
        return {
            "status": "healthy",
            "service": "student_loan_document_extractor",
            "version": "1.0.0",
            "processors": ["Form Parser", "Document OCR"],
            "features": [
                "Document upload and validation",
                "Batch processing",
                "Complete text extraction",
                "Table extraction with nested columns",
                "Loan comparison",
                "Storage and retrieval",
                "Processing status tracking"
            ],
            "storage_stats": stats
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }


@router.get("/accuracy")
async def get_accuracy_info():
    """
    Get information about accuracy calculation.
    
    Returns details about extraction accuracy metrics.
    """
    return {
        "accuracy_metrics": {
            "overall_accuracy": "Combined confidence from both processors",
            "form_parser_accuracy": "Form Parser confidence score",
            "ocr_accuracy": "Document OCR confidence score",
            "text_extraction_confidence": "Confidence for all text elements",
            "table_extraction_confidence": "Confidence for table data",
            "form_field_confidence": "Confidence for form fields",
            "page_confidences": "Per-page confidence scores",
            "low_confidence_items": "Items with confidence < 85%"
        },
        "processors": {
            "form_parser": "Best for structured forms and tables",
            "document_ocr": "Best for general text extraction"
        },
        "validation": "Both processors used for cross-validation",
        "target_accuracy": "F1 score >= 94% for key financial fields"
    }


@router.get("/api-info")
async def get_api_info():
    """
    Get API information and available endpoints.
    
    Returns comprehensive API documentation.
    """
    return {
        "api_version": "1.0.0",
        "service": "Student Loan Document Extractor Platform",
        "endpoints": {
            "upload": {
                "POST /api/v1/documents/upload": "Upload single document",
                "POST /api/v1/documents/batch-upload": "Upload multiple documents",
                "POST /api/v1/extract": "Upload and extract document"
            },
            "retrieval": {
                "GET /api/v1/documents/{id}": "Get document metadata",
                "GET /api/v1/documents/{id}/extracted-data": "Get extracted loan data",
                "GET /api/v1/loans": "Query loans with filters",
                "GET /api/v1/documents/{id}/status": "Get document processing status"
            },
            "comparison": {
                "POST /api/v1/compare": "Compare multiple loans"
            },
            "processing": {
                "GET /api/v1/processing-status/{job_id}": "Get batch job status"
            },
            "info": {
                "GET /api/v1/health": "Health check",
                "GET /api/v1/accuracy": "Accuracy information",
                "GET /api/v1/api-info": "API documentation"
            }
        },
        "supported_formats": ["PDF", "JPEG", "PNG", "TIFF"],
        "max_file_size": "50 MB",
        "max_pages": 50
    }


@router.post("/compare")
async def compare_loans(loan_data_list: list):
    """
    Compare multiple loan documents
    
    Accepts a list of loan data dictionaries and returns comparison results
    including best options by cost and flexibility, and pros/cons for each loan.
    """
    try:
        if len(loan_data_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 loans required for comparison"
            )
        
        logger.info(f"Comparing {len(loan_data_list)} loans")
        
        # Calculate metrics for each loan
        metrics_list = []
        for loan_data in loan_data_list:
            metrics = calculate_loan_metrics(loan_data)
            metrics_list.append(metrics)
        
        # Find best options
        best_by_cost = min(metrics_list, key=lambda m: m["total_cost_estimate"])
        best_by_flexibility = max(metrics_list, key=lambda m: m["flexibility_score"])
        
        # Generate comparison result
        result = {
            "loans": loan_data_list,
            "metrics": metrics_list,
            "best_by_cost": best_by_cost.get("loan_id", ""),
            "best_by_flexibility": best_by_flexibility.get("loan_id", ""),
            "comparison_notes": {
                "summary": f"Compared {len(loan_data_list)} loan options",
                "cost_range": f"Total cost ranges from ${min(m['total_cost_estimate'] for m in metrics_list):,.2f} to ${max(m['total_cost_estimate'] for m in metrics_list):,.2f}",
                "recommendation": "Review the comparison table and pros/cons for each loan"
            }
        }
        
        logger.info(f"Comparison complete - Best by cost: {result['best_by_cost']}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loans")
async def query_loans(
    loan_type: str = None,
    bank_name: str = None,
    min_amount: float = None,
    max_amount: float = None,
    skip: int = 0,
    limit: int = 100
):
    """
    Query loans with optional filters
    
    Supports filtering by loan type, bank name, and amount range.
    Returns paginated results.
    """
    try:
        logger.info(f"Querying loans with filters: type={loan_type}, bank={bank_name}, amount={min_amount}-{max_amount}")
        
        # In a real implementation, this would query a database
        # For now, return a sample response structure
        result = {
            "loans": [],
            "total": 0,
            "skip": skip,
            "limit": limit,
            "filters": {
                "loan_type": loan_type,
                "bank_name": bank_name,
                "min_amount": min_amount,
                "max_amount": max_amount
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_loan_metrics(loan_data: dict) -> dict:
    """
    Calculate comparison metrics for a loan
    
    Args:
        loan_data: Loan data dictionary
        
    Returns:
        Dictionary with calculated metrics
    """
    principal = loan_data.get("principal_amount", 0)
    interest_rate = loan_data.get("interest_rate", 0)
    tenure_months = loan_data.get("tenure_months", 0)
    processing_fee = loan_data.get("processing_fee", 0)
    moratorium_months = loan_data.get("moratorium_period_months", 0)
    
    # Calculate monthly EMI
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate > 0 and tenure_months > 0:
        monthly_emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    else:
        monthly_emi = principal / tenure_months if tenure_months > 0 else 0
    
    # Calculate total amounts
    total_amount = monthly_emi * tenure_months
    total_interest = total_amount - principal
    total_cost = total_amount + processing_fee
    effective_rate = (total_interest / principal * 100) if principal > 0 else 0
    
    # Calculate flexibility score (0-10)
    flexibility_score = 0.0
    
    # Moratorium period (up to 3 points)
    if moratorium_months > 0:
        flexibility_score += min(moratorium_months / 12 * 3, 3)
    
    # Prepayment penalty (3 points if no penalty)
    prepayment_penalty = loan_data.get("prepayment_penalty", "")
    if prepayment_penalty and ("no penalty" in prepayment_penalty.lower() or "nil" in prepayment_penalty.lower()):
        flexibility_score += 3
    
    # No collateral (2 points)
    if not loan_data.get("collateral_details"):
        flexibility_score += 2
    
    # No co-signer (2 points)
    if not loan_data.get("co_signer"):
        flexibility_score += 2
    
    return {
        "loan_id": loan_data.get("loan_id", loan_data.get("document_id", "unknown")),
        "total_cost_estimate": total_cost,
        "effective_interest_rate": effective_rate,
        "flexibility_score": flexibility_score,
        "monthly_emi": monthly_emi,
        "total_interest_payable": total_interest
    }
